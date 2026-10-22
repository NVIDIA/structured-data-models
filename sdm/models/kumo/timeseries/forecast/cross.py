# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any, Literal

import torch
from torch import Tensor


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
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            **factory_kwargs,
        )
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., C, N, D]
        *,
        batch_size_limit: int | Literal["auto"] | None = "auto",
    ) -> Tensor:  # [..., C, N, D]
        shape = x.transpose(-3, -2).shape
        x = x.transpose(-3, -2).flatten(0, -3)
        limit = 65_535 if batch_size_limit == "auto" else batch_size_limit
        limit = min(limit or 65_535, 65_535)
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            limit = max(x.size(0), 1)
        outputs = []
        for rows in x.split(limit):
            attended, _ = self.attn(rows, rows, rows, need_weights=True)
            outputs.append(self.norm(rows + attended))
        return torch.cat(outputs).reshape(shape).transpose(-3, -2)
