# Copyright (c) 2026 Mistral AI
# Licensed under Apache-2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor

from sdm.models.timestral.block import TimestralTransformerBlock
from sdm.nn import RMSNorm, RotaryEmbedding


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        column_every: int,
        num_heads: int,
        hidden_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.column_every = column_every

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=10_000,
            partial_rotary_factor=0.5,
            requires_grad=False,
            **factory_kwargs,
        )

        self.row_blocks = torch.nn.ModuleList(
            TimestralTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                hidden_channels=None
                if (i + 1) % column_every == 0
                else hidden_channels,
                rope=rope,
                **factory_kwargs,
            )
            for i in range(num_layers)
        )

        self.col_blocks = torch.nn.ModuleList(
            TimestralTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                **factory_kwargs,
            )
            for _ in range(num_layers // column_every)
        )

        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., C, R, D]
        row_mask: Tensor | None,  # [..., C, H, R, R]
        col_mask: Tensor | None,  # [..., R, H, C, C]
        pos: Tensor | None = None,  # [R] or [..., 1, R]
    ) -> Tensor:  # [..., C, R, D]
        for i, row_block in enumerate(self.row_blocks):
            x = row_block(
                query=x,
                attn_mask=row_mask,
                query_pos=pos,
                is_causal=row_mask is None,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            )

            if (i + 1) % self.column_every == 0:
                x = x.transpose(-3, -2)
                x = self.col_blocks[i // self.column_every](
                    query=x,
                    attn_mask=col_mask,
                    batch_size_limit="auto",
                    out=None if torch.is_grad_enabled() else x,
                ).transpose(-3, -2)

        return self.norm(x)
