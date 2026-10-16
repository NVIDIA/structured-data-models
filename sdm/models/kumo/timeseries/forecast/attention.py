# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor

from sdm.nn import Attention


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
        self.attn = Attention(
            channels=channels,
            num_query_heads=num_heads,
            dropout=dropout,
            **factory_kwargs,
        )
        # Preserve the upstream MultiheadAttention initialization.
        torch.nn.init.xavier_uniform_(self.attn.qkv_lin.weight)
        torch.nn.init.zeros_(self.attn.qkv_lin.bias)
        self.attn.out_lin.reset_parameters()
        torch.nn.init.zeros_(self.attn.out_lin.bias)
        self.norm = torch.nn.LayerNorm(channels, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Mix variates independently at every patch position.

        Args:
            x: Patch embeddings with shape ``[..., V, P, C]``, where ``V``
                is the number of variates, ``P`` is the number of patches,
                and ``C`` is the embedding width.

        Returns:
            Patch embeddings with shape ``[..., V, P, C]``.
        """
        x = x.transpose(-3, -2)  # [..., P, V, C]
        attended = self.attn(x)
        out = self.norm(x + self.dropout(attended))
        return out.transpose(-3, -2)
