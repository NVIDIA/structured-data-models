# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-native T5 encoder for continuous time-series embeddings."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, ModuleList, Parameter


class _RMSNorm(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        eps: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        self.weight = Parameter(
            torch.ones(channels, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # T5 accumulates variance in fp32, including for reduced precision.
        variance = x.float().square().mean(dim=-1, keepdim=True)
        x = x * (variance + self.eps).rsqrt()
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class _T5Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_heads: int,
        head_channels: int,
        dropout: float,
        eps: float,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.head_channels = head_channels
        self.attention_norm = _RMSNorm(channels, eps, **factory_kwargs)
        self.ffn_norm = _RMSNorm(channels, eps, **factory_kwargs)
        self.q = Linear(
            channels, num_heads * head_channels, bias=False, **factory_kwargs
        )
        self.k = Linear(
            channels, num_heads * head_channels, bias=False, **factory_kwargs
        )
        self.v = Linear(
            channels, num_heads * head_channels, bias=False, **factory_kwargs
        )
        self.o = Linear(
            num_heads * head_channels, channels, bias=False, **factory_kwargs
        )
        self.wi_0 = Linear(
            channels, hidden_channels, bias=False, **factory_kwargs
        )
        self.wi_1 = Linear(
            channels, hidden_channels, bias=False, **factory_kwargs
        )
        self.wo = Linear(
            hidden_channels, channels, bias=False, **factory_kwargs
        )
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        shape = (*x.shape[:-1], self.num_heads, self.head_channels)
        query = self.q(normalized).view(shape).transpose(-3, -2)
        key = self.k(normalized).view(shape).transpose(-3, -2)
        value = self.v(normalized).view(shape).transpose(-3, -2)
        # T5 uses unscaled dot products and shares position bias across layers.
        attended = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=bias.to(query.dtype),
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0,
        )
        attended = attended.transpose(-3, -2).flatten(-2)
        x = x + self.dropout(self.o(attended))
        normalized = self.ffn_norm(x)
        hidden = F.gelu(self.wi_0(normalized), approximate="tanh")
        hidden = hidden * self.wi_1(normalized)
        return x + self.dropout(self.wo(self.dropout(hidden)))


class T5Encoder(torch.nn.Module):
    """Bidirectional T5 encoder with gated GELU feed-forward layers.

    Accepts continuous patch embeddings; no vocabulary embedding or decoder is
    needed. Relative position bias is learned once and shared by all layers.

    Args:
        channels: Input and output embedding width.
        hidden_channels: Feed-forward hidden width.
        num_layers: Number of encoder blocks.
        num_heads: Number of attention heads.
        head_channels: Width of each attention head.
        num_buckets: Number of bidirectional relative-position buckets.
        max_distance: Distance at which relative-position buckets saturate.
        dropout: Attention, residual, and feed-forward dropout probability.
        eps: RMS normalization epsilon.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        head_channels: int = 64,
        num_buckets: int = 32,
        max_distance: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = Embedding(
            num_buckets, num_heads, **factory_kwargs
        )
        self.layers = ModuleList(
            [
                _T5Block(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    head_channels=head_channels,
                    dropout=dropout,
                    eps=eps,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = _RMSNorm(channels, eps, **factory_kwargs)
        self.dropout = Dropout(dropout)

    def _position_bias(self, length: int, device: torch.device) -> Tensor:
        position = torch.arange(length, device=device)
        distance = position.unsqueeze(0) - position.unsqueeze(1)
        half = self.num_buckets // 2
        exact = half // 2
        magnitude = distance.abs()
        logarithmic = (
            exact
            + (
                (magnitude.float().clamp_min(exact) / exact).log()
                * ((half - exact) / math.log(self.max_distance / exact))
            ).long()
        )
        bucket = torch.where(
            magnitude < exact, magnitude, logarithmic.clamp_max(half - 1)
        )
        bucket = bucket + (distance > 0).long() * half
        return (
            self.relative_attention_bias(bucket).permute(2, 0, 1).unsqueeze(0)
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode patches with shape ``[B, P, C]``.

        Args:
            x: Continuous patch embeddings.
            mask: Optional observed-patch mask of shape ``[B, P]``. False
                patches cannot be attended to as keys.

        Returns:
            Encoded patches of shape ``[B, P, C]``.
        """
        bias = self._position_bias(x.size(-2), x.device)
        if mask is not None:
            padding = x.new_zeros((*mask.shape[:-1], 1, 1, mask.size(-1)))
            padding = padding.masked_fill(
                ~mask.bool().unsqueeze(-2).unsqueeze(-2),
                torch.finfo(x.dtype).min,
            )
            bias = bias + padding
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, bias)
        return self.dropout(self.norm(x))
