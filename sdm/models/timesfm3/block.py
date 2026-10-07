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

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, RMSNorm, Sequential

from sdm.nn import RotaryEmbedding, SoftplusScale, TransformerBlock


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.mlp = Sequential(
            Linear(in_channels, out_channels, bias=bias, **factory_kwargs),
            ReLU(),
            Linear(out_channels, out_channels, bias=bias, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, bias, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.res(x)


class _AttentionBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp: torch.nn.Module | None,
        rope: RotaryEmbedding | None,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        head_dim = channels // num_heads
        query_transforms = Sequential(
            RMSNorm(head_dim, **factory_kwargs),
            SoftplusScale(
                head_dim,
                multiplier=1 / math.log(2),
                **factory_kwargs,
            ),
        )
        key_transform = Sequential(RMSNorm(head_dim, **factory_kwargs))
        if rope is not None:
            query_transforms.insert(0, rope)
            key_transform.insert(0, rope)

        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=RMSNorm(channels, **factory_kwargs),
            post_attn_norm=RMSNorm(channels, **factory_kwargs),
            query_transform=query_transforms,
            key_transform=key_transform,
            scale=1.0,
            bias=bias,
            **factory_kwargs,
        )


class TimesFM3TransformerBlock(torch.nn.Module):
    """TimesFM-3 transformer block from Jain and Sen (2026).

    Args:
        channels: Input, output, and feed-forward width.
        num_heads: Number of heads in each attention layer.
        device: Device on which to create parameters and buffers.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.time = _AttentionBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=None,
            rope=RotaryEmbedding(
                channels=channels // num_heads,
                layout="split_half",
                theta=10_000,
                requires_grad=False,
                device=device,
                dtype=torch.float32,
            ),
            bias=bias,
            **factory_kwargs,
        )
        self.var = _AttentionBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels, **factory_kwargs),
                Linear(channels, channels, bias=False, **factory_kwargs),
                ReLU(),
                Linear(channels, channels, bias=False, **factory_kwargs),
                RMSNorm(channels, **factory_kwargs),
            ),
            rope=None,
            bias=bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., V, N, C]
        patch_mask: Tensor | None = None,  # [..., V, N]
    ) -> Tensor:  # [..., V, N, C]
        """Apply causal time attention, variate attention, and an FFN.

        Args:
            x: Patch embeddings with shape ``[..., V, N, C]``.
            patch_mask: Boolean mask with shape ``[..., V, N]``. ``True``
                excludes a patch from attention keys. ``None`` means that
                no patches are excluded.

        Returns:
            Updated embeddings with shape ``[..., V, N, C]``.
        """
        if patch_mask is None:
            time_attn_mask = var_attn_mask = None
        else:
            N = x.size(-2)
            causal = torch.ones(
                (N, N), device=x.device, dtype=torch.bool
            ).tril()
            time_attn_mask = causal & ~patch_mask.unsqueeze(-2)
            var_attn_mask = (~patch_mask).transpose(-2, -1).unsqueeze(-2)

        hidden = self.time(
            query=x,
            attn_mask=time_attn_mask,
            is_causal=patch_mask is None,
        )
        return self.var(
            query=hidden.transpose(-3, -2), attn_mask=var_attn_mask
        ).transpose(-3, -2)
