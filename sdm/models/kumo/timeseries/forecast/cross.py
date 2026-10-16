# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any, Literal

import torch
from torch import Tensor

from sdm.nn import TransformerBlock


class CrossChannelAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.block = TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            mlp=None,
            **factory_kwargs,
        )
        torch.nn.init.xavier_uniform_(self.block.attn.qkv_lin.weight)
        torch.nn.init.zeros_(self.block.attn.qkv_lin.bias)
        self.block.attn.out_lin.reset_parameters()
        torch.nn.init.zeros_(self.block.attn.out_lin.bias)
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., C, N, D]
        *,
        batch_size_limit: int | Literal["auto"] | None = "auto",
    ) -> Tensor:  # [..., C, N, D]
        x = x.transpose(-3, -2)  # [..., P, V, C]
        x = self.block(
            query=x,
            batch_size_limit=batch_size_limit,
            out=None if torch.is_grad_enabled() else x,
        ).transpose(-3, -2)
        return self.norm(x)
