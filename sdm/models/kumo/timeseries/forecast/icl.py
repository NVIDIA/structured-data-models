# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleList

from sdm.models.kumo.timeseries.forecast.block import (
    KumoForecastTransformerBlock,
)
from sdm.nn import RMSNorm


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_layers: int,
        num_heads: int,
        num_buckets: int,
        max_distance: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        assert num_buckets >= 4
        assert max_distance > num_buckets // 4

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = Embedding(
            num_embeddings=num_buckets,
            embedding_dim=num_heads,
            **factory_kwargs,
        )
        self.layers = ModuleList(
            [
                KumoForecastTransformerBlock(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)

    def get_position_bias(self, length: int, device: torch.device) -> Tensor:
        num_buckets = self.num_buckets // 2  # Buckets per direction.
        max_exact = num_buckets // 2
        scale = num_buckets - max_exact
        scale /= math.log(self.max_distance / max_exact)

        pos = torch.arange(length, device=device)
        offset = pos.unsqueeze(0) - pos.unsqueeze(1)  # [P, P]
        dist = offset.abs()
        log_dist = (dist.float().clamp(min=max_exact) / max_exact).log()
        bucket = max_exact + (log_dist * scale).long()
        bucket = bucket.clamp_max(num_buckets - 1)
        bucket = torch.where(dist < max_exact, dist, bucket)
        bucket = bucket + (offset > 0) * num_buckets
        return self.relative_attention_bias(bucket).permute(2, 0, 1)

    def forward(
        self,
        x: Tensor,  # [..., N, D]
        *,
        batch_size_limit: int | Literal["auto"] | None = "auto",
    ) -> Tensor:  # [..., N, D]
        bias = self.get_position_bias(length=x.size(-2), device=x.device)
        for layer in self.layers:
            x = layer(
                query=x,
                attn_mask=bias,
                batch_size_limit=batch_size_limit,
                out=None if torch.is_grad_enabled() else x,
            )
        return self.norm(x)
