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

from sdm.models.tabfm.block import Encoder


class _RowInteraction(torch.nn.Module):
    """Apply TabFM v1.0.0 attention across the features of each row.

    Args:
        num_blocks: Number of row-attention blocks.
        channels: Number of channels in each feature embedding.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward layer.
        num_cls: Number of leading CLS tokens.
        output_full: Whether to return the full feature grid. If false, only
            the leading CLS tokens are returned and flattened.
        rope_theta: Rotary frequency base. ``None`` disables rotary embedding.
        row_chunk_size: Maximum number of independent rows processed at once.
            ``None`` disables chunking.
        device: Device on which to create parameters.
        dtype: Dtype of parameters.
    """

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_cls: int,
        output_full: bool = True,
        rope_theta: float | None = 100_000.0,
        row_chunk_size: int | None = 4096,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_cls <= 0:
            raise ValueError("num_cls must be positive")
        if row_chunk_size is not None and row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive or None")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_row = Encoder(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            hidden_channels=feedforward_channels,
            rope_theta=rope_theta,
            **factory_kwargs,
        )
        self.out_ln = torch.nn.RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.num_cls = num_cls
        self.output_full = output_full
        self.row_chunk_size = row_chunk_size

    def _stage(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        output = self.tf_row(x, attn_mask=attn_mask)
        if not self.output_full:
            output = output[:, : self.num_cls]
        return self.out_ln(output)

    def forward(
        self,
        x: Tensor,
        active_features: Tensor | None = None,
    ) -> Tensor:
        """Encode feature interactions independently for each table row.

        Args:
            x: Feature embeddings with shape ``[B, T, HC, E]``. The first
                ``num_cls`` columns are CLS tokens.
            active_features: Optional number of real features in each table,
                with shape ``[B]``. Only the CLS prefix and active feature
                prefix are visible as attention keys.

        Returns:
            ``[B, T, HC, E]`` when ``output_full`` is true, otherwise
            ``[B, T, num_cls * E]``.
        """
        batch_size, num_rows, num_columns, channels = x.shape
        if num_columns < self.num_cls:
            raise ValueError("x must contain the configured CLS token prefix")
        if active_features is not None and (
            active_features.shape != (batch_size,)
            or active_features.is_floating_point()
            or active_features.is_complex()
            or active_features.dtype == torch.bool
            or active_features.device != x.device
        ):
            raise ValueError(
                "active_features must be an integer [B] tensor on the "
                "input device"
            )
        # [B, T, HC, E] -> [B * T, HC, E]
        flattened = x.reshape(batch_size * num_rows, num_columns, channels)

        attn_mask = None
        if active_features is not None:
            active_columns = active_features.long() + self.num_cls
            column_index = torch.arange(num_columns, device=x.device)
            valid = column_index[None] < active_columns[:, None]
            attn_mask = valid.repeat_interleave(num_rows, dim=0)
            attn_mask = attn_mask[:, None]  # [B * T, 1, HC]

        if (
            self.row_chunk_size is None
            or flattened.size(0) <= self.row_chunk_size
        ):
            output = self._stage(flattened, attn_mask)
        else:
            output = torch.cat(
                [
                    self._stage(
                        flattened[start : start + self.row_chunk_size],
                        None
                        if attn_mask is None
                        else attn_mask[start : start + self.row_chunk_size],
                    )
                    for start in range(
                        0, flattened.size(0), self.row_chunk_size
                    )
                ],
                dim=0,
            )

        if self.output_full:
            return output.reshape(batch_size, num_rows, num_columns, channels)
        return output.reshape(batch_size, num_rows, self.num_cls * channels)
