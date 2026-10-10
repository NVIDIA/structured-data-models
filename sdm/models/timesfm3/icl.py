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

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList, ReLU, RMSNorm, Sequential

from sdm.models.timesfm3.block import TimesFM3TransformerBlock
from sdm.nn import RotaryEmbedding


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        num_layers: int,
        num_heads: int,
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
            **factory_kwargs,
        )

        self.time_blocks = ModuleList(
            TimesFM3TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                mlp=None,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.var_blocks = ModuleList(
            TimesFM3TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                mlp=Sequential(
                    RMSNorm(channels, **factory_kwargs),
                    Linear(channels, channels, bias=False, **factory_kwargs),
                    ReLU(),
                    Linear(channels, channels, bias=False, **factory_kwargs),
                    RMSNorm(channels, **factory_kwargs),
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.head = Linear(channels, out_channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., C, N, D]
        patch_mask: Tensor | None = None,  # [..., C, N]
    ) -> Tensor:  # [..., C, N, out_channels]

        if patch_mask is None:
            time_attn_mask = var_attn_mask = None
        else:
            time_attn_mask = torch.ones(
                (x.size(-2), x.size(-2)),
                device=x.device,
                dtype=torch.bool,
            ).tril() & ~patch_mask.unsqueeze(-2)
            var_attn_mask = ~patch_mask.transpose(-2, -1).unsqueeze(-2)

        for time_block, var_block in zip(self.time_blocks, self.var_blocks):
            x = time_block(
                query=x,
                attn_mask=time_attn_mask,
                is_causal=patch_mask is None,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            )
            x = x.transpose(-3, -2)
            x = var_block(
                query=x,
                attn_mask=var_attn_mask,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            ).transpose(-3, -2)

        return self.head(x)
