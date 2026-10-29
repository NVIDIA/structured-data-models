# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential

from sdm.nn import RMSNorm, RotaryEmbedding, TransformerBlock


class LayerNorm(RMSNorm):
    def forward(
        self, x: Tensor, rope: RotaryEmbedding | None = None
    ) -> Tensor:
        # T5 accumulates in float32 and rounds before multiplying half weights.
        if rope is not None:
            x = rope(x)
        variance = x.float().square().mean(-1, keepdim=True)
        x = x * (variance + (1e-6 if self.eps is None else self.eps)).rsqrt()
        if self.weight.dtype in {torch.float16, torch.bfloat16}:
            x = x.to(self.weight.dtype)
        return self.weight * x


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.mlp = Sequential(
            Linear(in_channels, hidden_channels, **factory_kwargs),
            ReLU(),
            Linear(hidden_channels, out_channels, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.res(x)


class Chronos2TransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        norm = LayerNorm(channels, eps=1e-6, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            query_norm=norm,
            key_value_norm=norm,
            query_transform=rope,
            key_transform=rope,
            mlp=None,
            scale=1.0,
            bias=False,
            **factory_kwargs,
        )
