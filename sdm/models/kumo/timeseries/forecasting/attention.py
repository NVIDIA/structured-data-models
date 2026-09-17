# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-variate attention for time-series forecasting."""

from typing import Any

import torch
from torch import Tensor


class CrossChannelAttention(torch.nn.Module):
    """Apply self-attention across variates at every patch position.

    Args:
        channels: Number of embedding channels.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.attn = torch.nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
            **factory_kwargs,
        )
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Mix variates independently at every patch position.

        Args:
            x: Patch embeddings with shape ``[B, V, P, C]``.

        Returns:
            Patch embeddings with shape ``[B, V, P, C]``.
        """
        batch_size, num_variates, num_patches, channels = x.size()
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_patches,
            num_variates,
            channels,
        )
        attended, _ = self.attn(x, x, x, need_weights=False)
        out = self.norm(x + self.dropout(attended))
        return out.reshape(
            batch_size,
            num_patches,
            num_variates,
            channels,
        ).permute(0, 2, 1, 3)
