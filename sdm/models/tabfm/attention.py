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

import math
from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleList

from sdm.models.tabfm.feedforward import (
    _ChunkedFeedForward,
    _SwiGLUFeedForward,
)
from sdm.nn import RotaryEmbedding, SoftplusScale
from sdm.nn import TransformerBlock as SDMTransformerBlock


def _attention_transforms(
    channels: int,
    num_heads: int,
    rope_theta: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Construct TabFM query and key head transformations."""
    head_channels = channels // num_heads
    factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    query: list[torch.nn.Module] = []
    key: list[torch.nn.Module] = []
    if rope_theta is not None:
        query.append(
            RotaryEmbedding(
                channels=head_channels,
                layout="interleaved",
                theta=rope_theta,
                requires_grad=False,
                **factory_kwargs,
            )
        )
        key.append(
            RotaryEmbedding(
                channels=head_channels,
                layout="interleaved",
                theta=rope_theta,
                requires_grad=False,
                **factory_kwargs,
            )
        )
    query.extend(
        [
            torch.nn.RMSNorm(head_channels, eps=1e-6, **factory_kwargs),
            SoftplusScale(
                head_channels,
                multiplier=1.442695041 / math.sqrt(head_channels),
                **factory_kwargs,
            ),
        ]
    )
    key.append(torch.nn.RMSNorm(head_channels, eps=1e-6, **factory_kwargs))
    return torch.nn.Sequential(*query), torch.nn.Sequential(*key)


def _feedforward_layer(
    channels: int,
    feedforward_channels: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    *,
    chunk_size: int | None = None,
    **_: Any,
) -> torch.nn.Module:
    """Construct the normalized TabFM SwiGLU feed-forward path."""
    factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    return _ChunkedFeedForward(
        module=torch.nn.Sequential(
            torch.nn.RMSNorm(channels, eps=1e-6, **factory_kwargs),
            _SwiGLUFeedForward(
                channels=channels,
                feedforward_channels=feedforward_channels,
                **factory_kwargs,
            ),
            torch.nn.RMSNorm(channels, eps=1e-6, **factory_kwargs),
        ),
        chunk_size=chunk_size,
    )


def _transformer_block(
    channels: int,
    num_heads: int,
    feedforward_channels: int,
    rope_theta: float | None,
    ffn_chunk_size: int | None,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> SDMTransformerBlock:
    query_transform, key_transform = _attention_transforms(
        channels=channels,
        num_heads=num_heads,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    return SDMTransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        feedforward_channels=feedforward_channels,
        norm=partial(torch.nn.RMSNorm, eps=1e-6),
        query_transform=query_transform,
        key_transform=key_transform,
        scale=1.0,
        device=device,
        dtype=dtype,
        shared_attention_norm=True,
        post_attention_norm=True,
        feedforward_layer=_feedforward_layer,
        feedforward_kwargs={"chunk_size": ffn_chunk_size},
    )


class _Encoder(torch.nn.Module):
    """Stack TabFM-configured transformer blocks."""

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        rope_theta: float | None = 100_000.0,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.blocks = ModuleList(
            _transformer_block(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                rope_theta=rope_theta,
                ffn_chunk_size=ffn_chunk_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        tensor: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply every transformer block."""
        for block in self.blocks:
            tensor = block(query=tensor, attn_mask=attn_mask)
        return tensor
