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


class CellEmbedder(torch.nn.Module):
    """Embed grouped numerical and categorical cells for TabFM v1.0.0.

    Feature groups use cyclic offsets ``2**index - 1``. Numerical and
    categorical slots have separate learned Fourier projections.

    Args:
        channels: Number of output channels per cell.
        feature_group_size: Number of cyclically shifted features per group.
        num_frequencies: Number of Fourier frequencies per group slot.
        device: Device on which to create parameters and buffers.
        dtype: Dtype of parameters and buffers.
    """

    fourier_frequencies: Tensor
    fourier_frequencies_cat: Tensor

    def __init__(
        self,
        channels: int,
        feature_group_size: int = 3,
        num_frequencies: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if (
            min(
                channels,
                feature_group_size,
                num_frequencies,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.feature_group_size = feature_group_size
        self.register_buffer(
            "fourier_frequencies",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "fourier_frequencies_cat",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                **factory_kwargs,
            ),
        )
        self.in_linear = Linear(
            2 * num_frequencies,
            channels,
            **factory_kwargs,
        )
        self.in_linear_cat = Linear(
            2 * num_frequencies,
            channels,
            **factory_kwargs,
        )

    def _group(self, x: Tensor, d: Tensor | None = None) -> Tensor:
        batch_size, num_rows, num_features = x.shape
        feature_index = torch.arange(num_features, device=x.device)
        offsets = (
            2
            ** torch.arange(
                self.feature_group_size,
                device=x.device,
            )
            - 1
        )
        if d is None:
            index = (feature_index[:, None] + offsets[None, :]) % num_features
            return x[..., index]

        safe_features = d.long().clamp_min(1)
        index = (
            feature_index[None, :, None] + offsets[None, None, :]
        ) % safe_features[:, None, None]
        index = index[:, None].expand(
            batch_size,
            num_rows,
            num_features,
            self.feature_group_size,
        )
        expanded = x.unsqueeze(-1).expand_as(index)
        return expanded.gather(dim=-2, index=index)

    def _embed(
        self,
        x: Tensor,
        cat_mask: Tensor | None,
        d: Tensor | None,
    ) -> Tensor:
        grouped = self._group(x, d=d).unsqueeze(-1).float()
        angles = grouped * self.fourier_frequencies.float()
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).to(x.dtype)
        numerical = self.in_linear(fourier)
        if cat_mask is None:
            return numerical.sum(dim=-2)

        angles = grouped * self.fourier_frequencies_cat.float()
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).to(x.dtype)
        categorical = self.in_linear_cat(fourier)
        grouped_mask = self._group(
            cat_mask[:, None].float(),
            d=d,
        ).bool()[..., None]
        return torch.where(grouped_mask, categorical, numerical).sum(dim=-2)

    def forward(
        self,
        x: Tensor,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        """Embed numerical and categorical cells.

        Args:
            x: Feature tensor with shape ``[B, T, H]``.
            cat_mask: Optional categorical mask with shape ``[B, H]``.
            d: Optional active feature counts with shape ``[B]``. Grouping
                wraps by these counts and padded output columns are zeroed.

        Returns:
            Cell embeddings with shape ``[B, T, H, E]``. ``E`` is
            ``channels``.
        """
        if x.dim() != 3 or not x.is_floating_point():
            raise ValueError("x must be a floating-point [B, T, H] tensor")
        batch_size, _, num_features = x.shape
        if cat_mask is not None and (
            cat_mask.shape != (batch_size, num_features)
            or cat_mask.dtype != torch.bool
        ):
            raise ValueError("cat_mask must be a boolean [B, H] tensor")
        if d is not None and (
            d.shape != (batch_size,) or d.is_floating_point()
        ):
            raise ValueError("d must be an integer [B] tensor")

        cell = self._embed(x=x, cat_mask=cat_mask, d=d)
        output = cell
        if d is not None:
            feature_index = torch.arange(num_features, device=x.device)
            active = feature_index[None, :] < d[:, None]
            output = output.masked_fill(~active[:, None, :, None], 0)
        return output
