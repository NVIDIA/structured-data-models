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
from torch.nn import Linear, ModuleList

from sdm.models.tabfm.block import TabFMTransformerBlock
from sdm.nn import InducedTransformerBlock


def _induced_block(
    channels: int,
    num_heads: int,
    feedforward_channels: int,
    num_inducing_points: int,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> InducedTransformerBlock:
    device = torch.device(device) if isinstance(device, str) else device
    return InducedTransformerBlock(
        channels=channels,
        num_inducing_points=num_inducing_points,
        inducing_block=TabFMTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            hidden_channels=feedforward_channels,
            device=device,
            dtype=dtype,
        ),
        output_block=TabFMTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            hidden_channels=feedforward_channels,
            device=device,
            dtype=dtype,
        ),
        device=device,
        dtype=dtype,
    )


class _ColumnEmbedding(torch.nn.Module):
    """Apply TabFM's distribution-aware embedding independently per column."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        col_chunk_size: int | None = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if col_chunk_size is not None and col_chunk_size <= 0:
            raise ValueError("col_chunk_size must be positive or None")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_col = ModuleList(
            _induced_block(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_inducing_points=num_inducing_points,
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )
        self.out_w = Linear(channels, channels, **factory_kwargs)
        self.ln_w = torch.nn.RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.col_chunk_size = col_chunk_size

    def _stage(self, src: Tensor, attn_mask: Tensor) -> Tensor:
        for block in self.tf_col:
            src = block(query=src, attn_mask=attn_mask)
        return self.ln_w(self.out_w(src))

    def forward(self, x: Tensor, context_size: Tensor) -> Tensor:
        """Embed each column using context rows as attention keys."""
        batch_size, num_rows, num_columns, channels = x.shape
        if (
            context_size.shape != (batch_size,)
            or context_size.is_floating_point()
            or context_size.is_complex()
            or context_size.dtype == torch.bool
            or context_size.device != x.device
        ):
            raise ValueError(
                "context_size must be integer [B] on the input device"
            )

        src = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_columns,
            num_rows,
            channels,
        )
        flat_context_size = context_size.repeat_interleave(num_columns)
        row_index = torch.arange(num_rows, device=x.device)
        attn_mask = row_index[None] < flat_context_size[:, None]

        if self.col_chunk_size is None or src.size(0) <= self.col_chunk_size:
            output = self._stage(src, attn_mask)
        else:
            output = torch.cat(
                [
                    self._stage(src_chunk, mask_chunk)
                    for src_chunk, mask_chunk in zip(
                        src.split(self.col_chunk_size, dim=0),
                        attn_mask.split(self.col_chunk_size, dim=0),
                        strict=True,
                    )
                ],
                dim=0,
            )

        return output.reshape(
            batch_size,
            num_columns,
            num_rows,
            channels,
        ).permute(0, 2, 1, 3)
