# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

from sdm.models.kumo.timeseries.anomaly.block import ResidualBlock
from sdm.models.kumo.timeseries.anomaly.embedding import DiffusionEmbedding


class DiffusionDenoiser(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        side_channels: int,
        num_steps: int,
        num_layers: int,
        num_heads: int,
        embedding_channels: int,
        input_channels: int = 2,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_steps = num_steps

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            channels=embedding_channels,
            **factory_kwargs,
        )
        self.strategy_embedding = Embedding(
            num_embeddings=2,
            embedding_dim=embedding_channels,
            **factory_kwargs,
        )
        self.input_projection = Linear(
            input_channels,
            channels,
            **factory_kwargs,
        )
        self.residual_layers = ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    side_channels=side_channels,
                    embedding_channels=embedding_channels,
                    num_heads=num_heads,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_after_skip = LayerNorm(channels, **factory_kwargs)
        self.output_projection1 = Linear(channels, channels, **factory_kwargs)
        self.norm_after_proj1 = LayerNorm(channels, **factory_kwargs)
        self.output_projection2 = Linear(channels, 1, **factory_kwargs)

        for lin in (self.input_projection, self.output_projection1):
            torch.nn.init.kaiming_normal_(
                tensor=lin.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            with torch.no_grad():
                lin.weight.mul_(0.5)
        torch.nn.init.xavier_uniform_(
            tensor=self.output_projection2.weight,
            gain=0.01,
        )

    def forward(
        self,
        x: Tensor,  # [B, I, K, L]
        side_info: Tensor,  # [B, S, K, L]
        step: Tensor,  # [B] or [1]
        strategy: Tensor,  # [B] or [1]
    ) -> Tensor:  # [B, K, L]
        x = self.input_projection(x.permute(0, 2, 3, 1)).relu()  # [B, K, L, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, K, L]

        diffusion_embedding = self.diffusion_embedding(step)
        strategy_embedding = self.strategy_embedding(strategy)

        skips = []
        for layer in self.residual_layers:
            x, skip = layer(
                x=x,
                side_info=side_info,
                diffusion_embedding=diffusion_embedding,
                strategy_embedding=strategy_embedding,
            )
            skips.append(skip)

        x = torch.stack(skips).sum(dim=0)  # [B, C, K, L]
        x = x / math.sqrt(len(skips))
        x = self.norm_after_skip(x.permute(0, 2, 3, 1))  # [B, K, L, C]
        x = self.output_projection1(x)
        x = self.norm_after_proj1(x).relu()
        return self.output_projection2(x).squeeze(-1)  # [B, K, L]
