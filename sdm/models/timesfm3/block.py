# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, RMSNorm, Sequential

from sdm.nn import RotaryEmbedding, SoftplusScale, TransformerBlock


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.mlp = Sequential(
            Linear(in_channels, out_channels, bias=False, **factory_kwargs),
            ReLU(),
            Linear(out_channels, out_channels, bias=False, **factory_kwargs),
        )
        self.res = Linear(in_channels, out_channels, False, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x) + self.res(x)


class TimesFM3TransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp: torch.nn.Module | None,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        query_transforms: list[torch.nn.Module] = [
            RMSNorm(channels // num_heads, **factory_kwargs),
            SoftplusScale(
                channels=channels // num_heads,
                multiplier=1 / math.log(2),
                **factory_kwargs,
            ),
        ]
        key_transforms: list[torch.nn.Module] = [
            RMSNorm(channels // num_heads, **factory_kwargs)
        ]
        if rope is not None:
            query_transforms = [rope, *query_transforms]
            key_transforms = [rope, *key_transforms]

        norm = RMSNorm(channels, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=norm,
            key_value_norm=norm,
            post_attn_norm=RMSNorm(channels, **factory_kwargs),
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            scale=1.0,
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
        factor = 13 if element_size <= 2 else 8
        return factor * length * element_size * self.attn.q_dim


if __name__ == "__main__":
    from sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: TimesFM3TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=None,
        ),
        channels_and_heads=[(1280, 16)],
    )
    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: TimesFM3TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels),
                Linear(channels, channels, bias=False),
                ReLU(),
                Linear(channels, channels, bias=False),
                RMSNorm(channels),
            ),
        ),
        channels_and_heads=[(1280, 16)],
    )
