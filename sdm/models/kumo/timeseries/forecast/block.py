# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Linear

from sdm.nn import TransformerBlock
from sdm.nn.attention import SDPA


class RMSNorm(torch.nn.RMSNorm):
    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + cast(float, self.eps))
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return self.weight * x


class Attention(SDPA):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> Tensor:
        assert seqused_key_value is None
        assert not is_causal
        query, key, value = (x.transpose(-3, -2) for x in (query, key, value))
        scores = query @ key.transpose(-1, -2)
        if attn_mask is not None:
            scores += attn_mask
        weights = scores.float().softmax(-1).to(scores.dtype)
        return (weights @ value).transpose(-3, -2)


class MLP(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.wi_0 = Linear(channels, hidden_channels, False, **factory_kwargs)
        self.wi_1 = Linear(channels, hidden_channels, False, **factory_kwargs)
        self.wo = Linear(hidden_channels, channels, False, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        hidden = self.wi_0(x)
        activated = (
            0.5
            * hidden
            * (
                1
                + (
                    math.sqrt(2 / math.pi)
                    * (hidden + 0.044715 * hidden.pow(3))
                ).tanh()
            )
        )
        x = activated * self.wi_1(x)
        return self.wo(x)


class KumoForecastTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=MLP(channels, hidden_channels, **factory_kwargs),
            query_norm=norm,
            key_value_norm=norm,
            scale=1.0,
            bias=False,
            **factory_kwargs,
        )
        self.attn.sdpa = Attention(num_heads, scale=1.0)
        self.hidden_channels = hidden_channels
        self.attn.out_lin.reset_parameters()

    def peak_bytes_per_example(
        self,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        length = max(query_length, key_value_length or 0)
        return math.ceil(11.25 * length * element_size * self.attn.q_dim)


if __name__ == "__main__":
    from sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: KumoForecastTransformerBlock(
            channels=channels,
            hidden_channels=2816,
            num_heads=num_heads,
        ),
        channels_and_heads=[(1024, 16)],
    )
