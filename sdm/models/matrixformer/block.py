# Copyright 2026 The MatrixFormer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm.nn import Attention, RMSNorm, RotaryEmbedding, SwiGLU


class AxialBlock(torch.nn.Module):
    r"""Axial transformer block over a grid of cells.

    The block attends across columns within each row, then across rows within
    each column, followed by a per-cell feed-forward layer.
    Each sublayer is pre-normalized and its residual is scaled by a learned
    scalar gain.

    Args:
        channels: The number of channels per cell.
        num_heads: The number of attention heads.
        hidden_channels: The hidden channels of the feed-forward layer.
        rotary_factor: The fraction of channels per head rotated by RoPE.
        eps: The epsilon of the RMS normalization layers.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        rotary_factor: float = 0.5,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_norm = RMSNorm(channels, eps=eps, **factory_kwargs)
        self.row_attn = _attention(
            channels, num_heads, rotary_factor, **factory_kwargs
        )
        self.col_norm = RMSNorm(channels, eps=eps, **factory_kwargs)
        self.col_attn = _attention(
            channels, num_heads, rotary_factor, **factory_kwargs
        )
        self.mlp_norm = RMSNorm(channels, eps=eps, **factory_kwargs)
        self.mlp = SwiGLU(channels, hidden_channels, **factory_kwargs)

        self.row_gain = Parameter(torch.ones(1, **factory_kwargs))
        self.col_gain = Parameter(torch.ones(1, **factory_kwargs))
        self.mlp_gain = Parameter(torch.ones(1, **factory_kwargs))

    def forward(
        self,
        x: Tensor,  # [..., R, C, F]
        row_pos: Tensor,  # [R]
        col_pos: Tensor,  # [C]
    ) -> Tensor:  # [..., R, C, F]
        r"""The forward pass.

        Args:
            x: The cell grid with shape ``[..., R, C, F]`` with ``R`` rows,
                ``C`` columns and ``F`` channels.
            row_pos: The RoPE positions of rows with shape ``[R]``.
            col_pos: The RoPE positions of columns with shape ``[C]``.
        """
        x = x + self.row_gain * self.row_attn(
            self.row_norm(x), query_pos=col_pos
        )

        h = self.col_norm(x).transpose(-3, -2)  # [..., C, R, F]
        h = self.col_attn(h, query_pos=row_pos).transpose(-3, -2)
        x = x + self.col_gain * h

        return x + self.mlp_gain * self.mlp(self.mlp_norm(x))


def _attention(
    channels: int,
    num_heads: int,
    rotary_factor: float,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> Attention:
    rope = RotaryEmbedding(
        channels=channels // num_heads,
        layout="interleaved",
        theta=10_000,
        requires_grad=False,
        partial_rotary_factor=rotary_factor,
        device=device,
        dtype=dtype,
    )
    return Attention(
        channels=channels,
        num_query_heads=num_heads,
        query_transform=rope,
        key_transform=rope,
        device=device,
        dtype=dtype,
    )
