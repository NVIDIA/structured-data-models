# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, ReLU, Sequential

from sdm.models.chronos2.block import Chronos2TransformerBlock
from sdm.nn import RMSNorm, RotaryEmbedding


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        num_heads: int,
        hidden_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=10_000,
            requires_grad=False,
            device=device,
        )
        self.time_blocks = ModuleList(
            Chronos2TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                mlp=None,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.group_blocks = ModuleList(
            Chronos2TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                mlp=Sequential(
                    RMSNorm(channels, eps=1e-6, **factory_kwargs),
                    Linear(channels, hidden_channels, False, **factory_kwargs),
                    ReLU(),
                    Linear(hidden_channels, channels, False, **factory_kwargs),
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., C, N, D]
        patch_mask: Tensor | None = None,  # [..., C, N]
    ) -> Tensor:  # [..., C, N, D]

        time_attn_mask = var_attn_mask = None
        if patch_mask is not None:
            time_attn_mask = ~patch_mask.unsqueeze(-2).unsqueeze(-2)
            var_attn_mask = time_attn_mask.transpose(-4, -1)

        for time_block, group_block in zip(
            self.time_blocks, self.group_blocks
        ):
            x = time_block(
                query=x,
                attn_mask=time_attn_mask,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            )
            x = x.transpose(-3, -2)
            x = group_block(
                query=x,
                attn_mask=var_attn_mask,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            ).transpose(-3, -2)

        return self.norm(x)
