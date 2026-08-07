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
from torch.nn import ModuleList, Parameter

from sdm.models.tabfm.attention import _MultiheadAttentionBlock


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
