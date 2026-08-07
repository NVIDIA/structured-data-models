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
from torch.nn import Linear, ModuleList, Parameter

from sdm.models.tabfm.attention import _MultiheadAttentionBlock, _RMSNorm


class _InducedSelfAttentionBlock(torch.nn.Module):
    """Apply checkpoint-compatible TabFM induced self-attention.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each SwiGLU feed-forward layer.
        num_inducing_points: Number of learned inducing vectors.
        ffn_chunk_size: Optional maximum number of flattened tokens processed
            by each feed-forward layer at once.
        device: Device on which to create parameters.
        dtype: Dtype of parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_inducing_points <= 0:
            raise ValueError("num_inducing_points must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.ind_vectors = Parameter(
            torch.zeros(num_inducing_points, channels, **factory_kwargs)
        )
        self.mab1 = _MultiheadAttentionBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            ffn_chunk_size=ffn_chunk_size,
            **factory_kwargs,
        )
        self.mab2 = _MultiheadAttentionBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            ffn_chunk_size=ffn_chunk_size,
            **factory_kwargs,
        )

    def forward(
        self,
        src: Tensor,  # [..., S, C]
        attn_mask: Tensor | None = None,  # [..., S]
    ) -> Tensor:  # [..., S, C]
        """Route input rows through the learned inducing vectors."""
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-2)  # [..., 1, S]
        hidden = self.mab1(
            query=self.ind_vectors,  # [M, C]
            key=src,
            value=src,
            attn_mask=attn_mask,
        )  # [..., M, C]
        return self.mab2(query=src, key=hidden, value=hidden)


class _SetTransformer(torch.nn.Module):
    """Stack checkpoint-compatible TabFM induced-attention blocks."""

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.blocks = ModuleList(
            _InducedSelfAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_inducing_points=num_inducing_points,
                ffn_chunk_size=ffn_chunk_size,
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        src: Tensor,  # [..., S, C]
        attn_mask: Tensor | None = None,  # [..., S]
    ) -> Tensor:  # [..., S, C]
        """Apply every induced-attention block in sequence."""
        for block in self.blocks:
            src = block(src=src, attn_mask=attn_mask)
        return src


class _ColumnEmbedding(torch.nn.Module):
    """Apply TabFM's distribution-aware embedding independently per column.

    Args:
        channels: Number of input and output channels per cell.
        num_blocks: Number of induced-attention blocks.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each SwiGLU feed-forward layer.
        num_inducing_points: Number of learned inducing vectors per block.
        col_chunk_size: Maximum number of flattened columns processed at once.
            ``None`` disables chunking.
        ffn_chunk_size: Optional maximum number of flattened feed-forward
            tokens processed at once.
        device: Device on which to create parameters.
        dtype: Dtype of parameters.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        col_chunk_size: int | None = 16,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if col_chunk_size is not None and col_chunk_size <= 0:
            raise ValueError("col_chunk_size must be positive or None")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_col = _SetTransformer(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            num_inducing_points=num_inducing_points,
            ffn_chunk_size=ffn_chunk_size,
            **factory_kwargs,
        )
        self.out_w = Linear(channels, channels, **factory_kwargs)
        self.ln_w = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.col_chunk_size = col_chunk_size

    def _stage(self, src: Tensor, attn_mask: Tensor) -> Tensor:
        return self.ln_w(self.out_w(self.tf_col(src, attn_mask=attn_mask)))

    def forward(
        self,
        x: Tensor,  # [B, T, H, E]
        context_size: Tensor,  # [B]
    ) -> Tensor:  # [B, T, H, E]
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

        # [B, T, H, E] -> [B * H, T, E]
        src = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_columns,
            num_rows,
            channels,
        )
        flat_context_size = context_size.repeat_interleave(num_columns)
        row_index = torch.arange(num_rows, device=x.device)
        attn_mask = row_index[None] < flat_context_size[:, None]  # [B * H, T]

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

        # [B * H, T, E] -> [B, T, H, E]
        return output.reshape(
            batch_size,
            num_columns,
            num_rows,
            channels,
        ).permute(0, 2, 1, 3)
