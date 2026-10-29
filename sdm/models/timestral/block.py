# Copyright (c) 2026 Mistral AI
# Licensed under Apache-2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Sigmoid, SiLU

from sdm.nn import RMSNorm, RotaryEmbedding, SwiGLU, TransformerBlock


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.mlp = Sequential(
            Linear(in_channels, hidden_channels, bias, **factory_kwargs),
            SiLU(),
            Linear(hidden_channels, out_channels, bias, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, bias, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.res(x)


class MonotoneQuantiles(torch.nn.Module):
    width: Tensor

    def __init__(
        self,
        levels: Sequence[float],
        width_factor: float = 3.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.width_factor = width_factor
        self.median_index = list(levels).index(0.5)

        widths = [
            levels[i + 1] - levels[i]
            if i < self.median_index
            else levels[i] - levels[i - 1]
            if i > self.median_index
            else 0.0
            for i in range(len(levels))
        ]
        self.register_buffer(
            "width",
            torch.tensor(widths, device=device, dtype=dtype),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        m = self.median_index
        median = x[..., m : m + 1]

        inc = self.width_factor * self.width.to(x) * F.softplus(x)
        up = inc[..., m + 1 :].cumsum(-1)
        down = inc[..., :m].flip(-1).cumsum(-1).flip(-1)
        return torch.cat([median - down, median, median + up], dim=-1)


class TimestralTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int | None,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            query_norm=norm,
            key_value_norm=norm,
            query_transform=rope,
            key_transform=rope,
            gate=Sequential(
                Linear(channels, channels, bias=False, **factory_kwargs),
                Sigmoid(),
            ),
            mlp=Sequential(
                RMSNorm(channels, eps=1e-6, **factory_kwargs),
                SwiGLU(
                    channels,
                    hidden_channels,
                    bias=False,
                    **factory_kwargs,
                ),
            )
            if hidden_channels is not None
            else None,
            bias=False,
            **factory_kwargs,
        )

    def peak_bytes_per_example(
        self,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        length = max(query_length, key_value_length or 0)
        factor = 8 if self.mlp is None else 12
        return factor * length * element_size * self.attn.q_dim


if __name__ == "__main__":
    from sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: TimestralTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            hidden_channels=None,
        ),
        channels_and_heads=[(768, 12)],
    )
    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: TimestralTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            hidden_channels=2048,
        ),
        channels_and_heads=[(768, 12)],
    )
