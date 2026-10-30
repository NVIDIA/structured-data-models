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
        x: Tensor,  # [S, N, C]
        observed: Tensor,  # [S, N]
        group_ids: Tensor,  # [S]
    ) -> Tensor:
        # Finite additive masks preserve upstream's fully masked-row behavior.
        minimum = torch.finfo(x.dtype).min
        time_mask = (~observed).to(x.dtype)[:, None, None, :] * minimum
        group_mask = group_ids[:, None] == group_ids[None, :]
        group_mask = group_mask[..., None] & observed[None, :, :]
        group_mask = (~group_mask).permute(2, 0, 1).unsqueeze(1)
        group_mask = group_mask.to(x.dtype) * minimum
        for time_block, group_block in zip(
            self.time_blocks, self.group_blocks
        ):
            x = time_block(query=x, attn_mask=time_mask)
            x = group_block(
                query=x.transpose(0, 1),
                attn_mask=group_mask,
            ).transpose(0, 1)
        return self.norm(x)
