# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
from torch import Tensor


def _conv1d(
    in_channels: int,
    out_channels: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Conv1d:
    layer = torch.nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        device=device,
        dtype=dtype,
    )
    torch.nn.init.kaiming_normal_(
        tensor=layer.weight, mode="fan_out", nonlinearity="relu"
    )
    with torch.no_grad():
        layer.weight.mul_(0.5)
    return layer


class ResidualBlock(torch.nn.Module):
    """Kumo-Anomaly temporal/feature attention with gated residual outputs.

    Args:
        channels: Input and output width.
        side_channels: Width of the precomputed conditioning information.
        embedding_channels: Width of diffusion and masking-strategy embeddings.
        num_heads: Attention heads for each axis.
        hidden_channels: Transformer feedforward width.
        dropout: Transformer attention and feedforward dropout probability.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        side_channels: int,
        embedding_channels: int,
        num_heads: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.diffusion_projection = torch.nn.Linear(
            in_features=embedding_channels,
            out_features=channels,
            **factory_kwargs,
        )
        self.strategy_projection = torch.nn.Linear(
            in_features=embedding_channels,
            out_features=channels,
            **factory_kwargs,
        )
        torch.nn.init.xavier_uniform_(
            tensor=self.diffusion_projection.weight, gain=0.5
        )
        torch.nn.init.xavier_uniform_(
            tensor=self.strategy_projection.weight, gain=0.5
        )
        self.cond_projection = _conv1d(
            in_channels=side_channels,
            out_channels=2 * channels,
            **factory_kwargs,
        )
        self.mid_projection = _conv1d(channels, 2 * channels, **factory_kwargs)
        self.output_projection = _conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            **factory_kwargs,
        )

        # Native post-norm blocks preserve the released attention dropout and
        # residual normalization order; SDM's block uses a different ordering.
        self.time_layer = torch.nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            **factory_kwargs,
        )
        self.feature_layer = torch.nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            **factory_kwargs,
        )
        self.norm_after_time = torch.nn.LayerNorm(channels, **factory_kwargs)
        self.norm_after_feature = torch.nn.LayerNorm(
            normalized_shape=channels, **factory_kwargs
        )
        self.norm_after_gate = torch.nn.LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [B, C, K, L]
        side_info: Tensor,  # [B, S, K, L]
        diffusion_embedding: Tensor,  # [B or 1, E]
        strategy_embedding: Tensor,  # [B or 1, E]
    ) -> tuple[Tensor, Tensor]:
        """Return residual and skip tensors with the same shape as ``x``."""
        batch, channels, features, length = x.shape
        y = x.permute(0, 2, 3, 1)  # [B, K, L, C]
        y = y + self.diffusion_projection(diffusion_embedding)[:, None, None]
        y = y + self.strategy_projection(strategy_embedding)[:, None, None]
        # The checkpoint bypasses both attention and normalization on singleton
        # axes, which is not equivalent to applying single-token attention.
        if length > 1:
            y = y.reshape(batch * features, length, channels)
            y = self.norm_after_time(self.time_layer(y))
            y = y.reshape(batch, features, length, channels)
        if features > 1:
            y = y.transpose(1, 2).reshape(batch * length, features, channels)
            y = self.norm_after_feature(self.feature_layer(y))
            y = y.reshape(batch, length, features, channels).transpose(1, 2)

        y = y.permute(0, 3, 1, 2).flatten(2)
        y = self.mid_projection(y) + self.cond_projection(side_info.flatten(2))
        gate, value = y.chunk(2, dim=1)
        y = gate.sigmoid() * value.tanh()
        y = self.norm_after_gate(y.transpose(1, 2)).transpose(1, 2)
        residual, skip = (
            self.output_projection(y)
            .reshape(batch, 2 * channels, features, length)
            .chunk(2, dim=1)
        )
        return (x + residual) / math.sqrt(2.0), skip
