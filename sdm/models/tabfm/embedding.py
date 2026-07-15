# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the structured-data-models package.

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear


class _CellEmbedder(torch.nn.Module):
    """Embed TabFM v1.0.0 feature cells.

    This checkpoint-specific operator accepts an explicit categorical mask, so
    features may appear in any order after preprocessing and column shuffling.

    Args:
        embed_dim: Number of output channels per cell.
        feature_group_size: Number of cyclically shifted features per group.
        num_frequencies: Number of Fourier frequencies per group slot.
        row_chunk_size: Maximum number of rows expanded at once. ``None``
            disables chunking.
        device: Device on which to create parameters and buffers.
        dtype: Compute dtype for the learned projections. Fourier-frequency
            buffers remain in float32.
    """

    fourier_frequencies: Tensor
    fourier_frequencies_cat: Tensor

    def __init__(
        self,
        embed_dim: int,
        feature_group_size: int = 3,
        num_frequencies: int = 32,
        row_chunk_size: int | None = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.feature_group_size = feature_group_size
        self.row_chunk_size = row_chunk_size
        self.register_buffer(
            "fourier_frequencies",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                device=device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "fourier_frequencies_cat",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                device=device,
                dtype=torch.float32,
            ),
        )
        self.in_linear = Linear(
            2 * num_frequencies,
            embed_dim,
            **factory_kwargs,
        )
        self.in_linear_cat = Linear(
            2 * num_frequencies,
            embed_dim,
            **factory_kwargs,
        )

    def _group(
        self,
        features: Tensor,
        active_features: Tensor | None,
    ) -> Tensor:
        """Apply cyclic feature offsets to ``[B, T, H]`` input."""
        batch_size, num_rows, num_features = features.shape
        feature_index = torch.arange(num_features, device=features.device)
        offsets = (
            2
            ** torch.arange(
                self.feature_group_size,
                device=features.device,
            )
            - 1
        )
        if active_features is None:
            index = (feature_index[:, None] + offsets[None, :]) % num_features
            return features[..., index]

        safe_features = active_features.long().clamp_min(1)
        index = (
            feature_index[None, :, None] + offsets[None, None, :]
        ) % safe_features[:, None, None]
        index = index[:, None].expand(
            batch_size,
            num_rows,
            num_features,
            self.feature_group_size,
        )
        expanded = features.unsqueeze(-1).expand_as(index)
        return expanded.gather(dim=-2, index=index)

    def _embed_rows(
        self,
        features: Tensor,
        categorical_mask: Tensor | None,
        active_features: Tensor | None,
    ) -> Tensor:
        grouped = self._group(features, active_features).unsqueeze(-1).float()

        angles = grouped * self.fourier_frequencies
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1)
        numerical = self.in_linear(fourier.to(features.dtype))
        if categorical_mask is None:
            return numerical.sum(dim=-2)

        angles_cat = grouped * self.fourier_frequencies_cat
        fourier_cat = torch.cat(
            [angles_cat.sin(), angles_cat.cos()],
            dim=-1,
        )
        categorical = self.in_linear_cat(fourier_cat.to(features.dtype))
        grouped_mask = self._group(
            categorical_mask[:, None],
            active_features,
        )[..., None]
        return torch.where(grouped_mask, categorical, numerical).sum(dim=-2)

    def forward(
        self,
        features: Tensor,
        categorical_mask: Tensor | None = None,
        active_features: Tensor | None = None,
    ) -> Tensor:
        """Embed feature values.

        Args:
            features: Dense feature values with shape ``[B, T, H]``. Values
                must already use the projection's compute dtype.
            categorical_mask: Optional boolean mask with shape ``[B, H]``.
                Each grouped slot uses the projection for its source feature.
            active_features: Optional integer tensor with shape ``[B]``. Each
                member wraps grouping over its active prefix; padded output
                columns are zero.

        Returns:
            Cell embeddings with shape ``[B, T, H, E]``.
        """
        batch_size, num_rows, num_features = features.shape
        if self.row_chunk_size is None or num_rows <= self.row_chunk_size:
            output = self._embed_rows(
                features,
                categorical_mask,
                active_features,
            )
        else:
            # Rows are independent here, so chunking bounds the Fourier tensor.
            output = torch.cat(
                [
                    self._embed_rows(
                        features[:, start : start + self.row_chunk_size],
                        categorical_mask,
                        active_features,
                    )
                    for start in range(0, num_rows, self.row_chunk_size)
                ],
                dim=1,
            )

        if active_features is None:
            return output

        feature_index = torch.arange(num_features, device=features.device)
        active = feature_index[None, :] < active_features[:, None]
        return output.masked_fill(~active[:, None, :, None], 0)
