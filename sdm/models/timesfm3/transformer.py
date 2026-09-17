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

import torch
from torch import Tensor


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: Tensor,
    query_index_offset: Tensor | None = None,
    kv_length: int = 0,
    causal: bool = True,
) -> Tensor:
    """Create an attention mask in which ``True`` permits attention.

    Args:
        query_length: Number of query positions ``Q``.
        num_all_masked_kv: Number of leading masked key/value positions with
            shape ``[B]``.
        query_index_offset: Optional query-position offsets with shape ``[B]``
            for cached decoding.
        kv_length: Number of key/value positions ``K``. A value of zero uses
            ``query_length``.
        causal: Whether queries may attend only to preceding positions.

    Returns:
        Boolean mask with shape ``[B, 1, Q, K]`` when causal and broadcastable
        shape ``[B, 1, 1, K]`` otherwise.
    """
    if kv_length == 0:
        kv_length = query_length

    query_index = torch.arange(
        query_length,
        device=num_all_masked_kv.device,
    ).view(1, 1, -1, 1)
    if query_index_offset is not None:
        query_index = query_index + query_index_offset.view(-1, 1, 1, 1)
    kv_index = torch.arange(
        kv_length,
        device=num_all_masked_kv.device,
    ).view(1, 1, 1, -1)
    mask = kv_index >= num_all_masked_kv.view(-1, 1, 1, 1)
    if causal:
        return (query_index >= kv_index) & mask
    return mask


def make_segment_mask(segment_ids: Tensor) -> Tensor:
    """Create an attention mask that isolates packed segments.

    Args:
        segment_ids: Segment identifiers with shape ``[B, S]``.

    Returns:
        Boolean mask with shape ``[B, 1, S, S]``.
    """
    return (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)


class RotaryPositionalEmbedding(torch.nn.Module):
    """Apply RoPE from the `RoFormer paper <https://arxiv.org/abs/2104.09864>`_.

    Args:
        embedding_dims: Size of the last input dimension.
        min_timescale: Minimum rotation timescale.
        max_timescale: Maximum rotation timescale.
        device: Device on which to create the timescale buffer.
    """

    timescale: Tensor

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

        half_dim = embedding_dims // 2
        fraction = (
            2.0
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / embedding_dims
        )
        timescale = min_timescale * (max_timescale / min_timescale) ** fraction
        self.register_buffer("timescale", timescale, persistent=False)

    def forward(
        self,
        inputs: Tensor,
        position: Tensor | None = None,
    ) -> Tensor:
        """Apply rotary positional embeddings.

        Args:
            inputs: Input with shape ``[B, S, D]`` or ``[B, S, H, D]``.
            position: Optional positions with shape ``[B, S]``. Sequential
                positions are used by default.

        Returns:
            Rotated input with the same shape as ``inputs``.
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must "
                "match the hidden dimension of the inputs."
            )

        if position is None:
            position = torch.arange(
                inputs.shape[1],
                device=inputs.device,
                dtype=torch.float32,
            ).unsqueeze(0)

        if inputs.dim() == 4:
            position = position.unsqueeze(-1).unsqueeze(-1)
            timescale = self.timescale.view(1, 1, 1, -1)
        elif inputs.dim() == 3:
            position = position.unsqueeze(-1)
            timescale = self.timescale.view(1, 1, -1)
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid = position.float() / timescale
        sin = sinusoid.sin()
        cos = sinusoid.cos()
        first_half, second_half = inputs.chunk(2, dim=-1)
        first = first_half * cos - second_half * sin
        second = second_half * cos + first_half * sin
        return torch.cat((first, second), dim=-1)
