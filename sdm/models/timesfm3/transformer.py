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


def make_attn_mask(patch_mask: Tensor, causal: bool = True) -> Tensor:
    """Create an attention mask in which ``True`` permits attention.

    Args:
        patch_mask: Masked patches with shape ``[B, N]``.
        causal: Whether queries may attend only to preceding positions.

    Returns:
        Boolean mask with shape ``[B, 1, N, N]`` when causal and broadcastable
        shape ``[B, 1, 1, N]`` otherwise.
    """
    mask = ~patch_mask[:, None, None, :]
    if not causal:
        return mask
    causal_mask = torch.ones(
        patch_mask.size(1),
        patch_mask.size(1),
        dtype=torch.bool,
        device=patch_mask.device,
    ).tril()
    return causal_mask[None, None] & mask


class RotaryPositionalEmbedding(torch.nn.Module):
    """Apply RoPE from the `RoFormer paper <https://arxiv.org/abs/2104.09864>`_.

    Args:
        embedding_dims: Size of the last input dimension.
        device: Device on which to create the timescale buffer.
    """

    timescale: Tensor

    def __init__(
        self,
        embedding_dims: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.register_buffer(
            "timescale",
            self._make_timescale(device),
            persistent=False,
        )

    def _make_timescale(self, device: torch.device | str | None) -> Tensor:
        half_dim = self.embedding_dims // 2
        fraction = (
            2.0
            * torch.arange(half_dim, dtype=torch.float32, device=device)
            / self.embedding_dims
        )
        return 10_000**fraction

    def _reset_timescale(self, device: torch.device | str) -> None:
        self.timescale = self._make_timescale(device)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply rotary positional embeddings.

        Args:
            inputs: Input with shape ``[B, S, D]`` or ``[B, S, H, D]``.

        Returns:
            Rotated input with the same shape as ``inputs``.
        """
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must "
                "match the hidden dimension of the inputs."
            )

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
        sin = sinusoid.sin().to(inputs.dtype)
        cos = sinusoid.cos().to(inputs.dtype)
        first_half, second_half = inputs.chunk(2, dim=-1)
        first = first_half * cos - second_half * sin
        second = second_half * cos + first_half * sin
        return torch.cat((first, second), dim=-1)
