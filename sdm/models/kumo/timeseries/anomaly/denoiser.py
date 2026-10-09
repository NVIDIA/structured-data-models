# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Conv1d, Embedding, LayerNorm, ModuleList

from sdm.models.kumo.timeseries.anomaly.block import ResidualBlock
from sdm.models.kumo.timeseries.anomaly.embedding import DiffusionEmbedding


class DiffusionDenoiser(torch.nn.Module):
    """Predict noise for one Kumo-Anomaly diffusion step.

    Callers supply normalized noisy/observed inputs and precomputed side
    information. This component does not sample or compute anomaly scores.

    Args:
        channels: Width of the residual blocks.
        side_channels: Width of the precomputed conditioning information.
        num_steps: Number of discrete diffusion steps.
        num_layers: Number of residual blocks, at least one.
        num_heads: Attention heads for each axis.
        embedding_channels: Even diffusion and strategy embedding width,
            at least four.
        input_channels: Input width (two for conditional, one unconditional).
        hidden_channels: Transformer feedforward width.
        dropout: Transformer attention and feedforward dropout probability.
        device: The device.
        dtype: The parameter and output dtype.
    """

    def __init__(
        self,
        channels: int,
        side_channels: int,
        num_steps: int,
        num_layers: int,
        num_heads: int,
        embedding_channels: int = 128,
        input_channels: int = 2,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
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
        self.input_projection = Conv1d(
            in_channels=input_channels,
            out_channels=channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.output_projection1 = Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.output_projection2 = Conv1d(
            in_channels=channels,
            out_channels=1,
            kernel_size=1,
            **factory_kwargs,
        )
        for projection in (self.input_projection, self.output_projection1):
            torch.nn.init.kaiming_normal_(
                tensor=projection.weight, mode="fan_out", nonlinearity="relu"
            )
            with torch.no_grad():
                projection.weight.mul_(0.5)
        torch.nn.init.xavier_uniform_(
            tensor=self.output_projection2.weight, gain=0.01
        )
        self.residual_layers = ModuleList(
            ResidualBlock(
                channels=channels,
                side_channels=side_channels,
                embedding_channels=embedding_channels,
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                dropout=dropout,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm_after_skip = LayerNorm(channels, **factory_kwargs)
        self.norm_after_proj1 = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        side_info: Tensor,
        step: Tensor,
        strategy: Tensor,
    ) -> Tensor:
        """Predict noise for normalized time-series windows.

        Args:
            x: Input with shape ``[B, I, K, L]``, where ``B`` is batch size,
                ``I`` input channels, ``K`` features, and ``L`` time steps.
            side_info: Conditioning with shape ``[B, S, K, L]``, where
                ``S`` is ``side_channels``.
            step: Integer diffusion indices in ``[0, num_steps)`` with
                shape ``[B]`` or ``[1]``.
            strategy: Integer masking-strategy indices in ``{0, 1}`` with
                shape ``[B]`` or ``[1]``. Single-entry indices broadcast
                across the batch independently for either embedding.

        Returns:
            Predicted noise with shape ``[B, K, L]`` in input feature/time
            order. No denormalization or mask restoration is applied.
        """
        batch, _, features, length = x.shape
        x = self.input_projection(x.flatten(2)).relu()
        x = x.reshape(batch, -1, features, length)
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
        x = torch.stack(skips).sum(dim=0) / math.sqrt(len(skips))
        x = self.norm_after_skip(x.flatten(2).transpose(1, 2))  # [B, K * L, C]
        x = self.output_projection1(x.transpose(1, 2))
        x = self.norm_after_proj1(x.transpose(1, 2)).transpose(1, 2)
        return self.output_projection2(x.relu()).reshape(
            batch, features, length
        )
