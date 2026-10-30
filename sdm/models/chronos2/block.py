# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# ruff: noqa: D101

from typing import Any

import torch
from torch.nn import Linear, ReLU, Sequential

from sdm.nn import RMSNorm, RotaryEmbedding, TransformerBlock


class Chronos2TransformerBlock(TransformerBlock):
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
        norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            query_norm=norm,
            key_value_norm=norm,
            query_transform=rope,
            key_transform=rope,
            mlp=mlp,
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
        factor = 7 if self.mlp is None else 10
        return factor * length * element_size * self.attn.q_dim


if __name__ == "__main__":
    from sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: Chronos2TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=None,
        ),
        channels_and_heads=[(768, 12)],
    )
    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: Chronos2TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels, eps=1e-6),
                Linear(channels, 3072, bias=False),
                ReLU(),
                Linear(3072, channels, bias=False),
            ),
        ),
        channels_and_heads=[(768, 12)],
    )
