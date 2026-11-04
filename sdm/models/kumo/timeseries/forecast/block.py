# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from sdm.nn import RMSNorm, TransformerBlock


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
        x = F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x)
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
