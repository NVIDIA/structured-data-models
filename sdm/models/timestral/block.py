# Copyright (c) 2026 Mistral AI
# Licensed under Apache-2.0; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101

from typing import Any

import torch
from torch.nn import Linear, Sequential, Sigmoid

from sdm.nn import RMSNorm, RotaryEmbedding, SwiGLU, TransformerBlock


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
