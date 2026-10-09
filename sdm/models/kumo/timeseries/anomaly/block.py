# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Conv1d, LayerNorm, Linear, TransformerEncoderLayer


class ResidualBlock(torch.nn.Module):
    """Kumo-Anomaly temporal/feature attention with gated residual outputs.

    Time and feature axes of length one bypass their attention and subsequent
    normalization, matching the upstream diffusion block.

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
        self.diffusion_projection = Linear(
            in_features=embedding_channels,
            out_features=channels,
            **factory_kwargs,
        )
        self.strategy_projection = Linear(
            in_features=embedding_channels,
            out_features=channels,
            **factory_kwargs,
        )
        for projection in (
            self.diffusion_projection,
            self.strategy_projection,
        ):
            torch.nn.init.xavier_uniform_(projection.weight, gain=0.5)

        self.cond_projection = Conv1d(
            in_channels=side_channels,
            out_channels=2 * channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.mid_projection = Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.output_projection = Conv1d(
            in_channels=channels,
            out_channels=2 * channels,
            kernel_size=1,
            **factory_kwargs,
        )
        for projection in (
            self.cond_projection,
            self.mid_projection,
            self.output_projection,
        ):
            torch.nn.init.kaiming_normal_(
                tensor=projection.weight, mode="fan_out", nonlinearity="relu"
            )
            with torch.no_grad():
                projection.weight.mul_(0.5)

        # Preserve post-residual normalization and attention dropout, which
        # SDM's TransformerBlock does not currently provide together.
        self.time_layer = TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            **factory_kwargs,
        )
        self.feature_layer = TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            **factory_kwargs,
        )
        self.norm_after_time = LayerNorm(channels, **factory_kwargs)
        self.norm_after_feature = LayerNorm(channels, **factory_kwargs)
        self.norm_after_gate = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        side_info: Tensor,
        diffusion_embedding: Tensor,
        strategy_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the updated residual stream and skip connection.

        Args:
            x: Hidden features with shape ``[B, C, K, L]``, where ``B`` is
                batch size, ``C`` channels, ``K`` features, and ``L`` time
                steps.
            side_info: Conditioning information with shape ``[B, S, K, L]``,
                where ``S`` is ``side_channels``.
            diffusion_embedding: Diffusion-step embeddings with shape
                ``[B, E]`` or ``[1, E]``, with ``E = embedding_channels``.
            strategy_embedding: Masking-strategy embeddings with shape
                ``[B, E]`` or ``[1, E]``. Single-row embeddings are broadcast
                across the batch.

        Returns:
            Residual and skip tensors, each with the same shape as ``x``.
        """
        batch, channels, features, length = x.shape
        y = x.permute(0, 2, 3, 1)  # [B, K, L, C]
        y = y + self.diffusion_projection(diffusion_embedding)[:, None, None]
        y = y + self.strategy_projection(strategy_embedding)[:, None, None]
        if length > 1:
            y = y.reshape(batch * features, length, channels)  # [B * K, L, C]
            y = self.norm_after_time(self.time_layer(y))
            y = y.reshape(batch, features, length, channels)
        if features > 1:
            y = y.transpose(1, 2).reshape(
                batch * length, features, channels
            )  # [B * L, K, C]
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
