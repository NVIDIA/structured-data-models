# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor


class SinusoidalEmbedding(torch.nn.Module):
    r"""Sinusoidal positional embedding from the
    `"Attention Is All You Need" <https://arxiv.org/abs/1706.03762>`_ paper.

    Args:
        theta: The base used to compute positional frequencies.
    """  # noqa: D205

    def __init__(self, theta: float = 10_000) -> None:
        super().__init__()
        self.theta = theta

    def forward(self, x: Tensor, pos: Tensor | None = None) -> Tensor:
        """The forward pass.

        Args:
            x: Tensor with shape ``[..., S, C]``, where ``S`` is the sequence
                length and ``C`` is the number of channels.
            pos: Optional position indices with shape ``[..., S]``.
        """
        *_, S, C = x.size()
        if pos is None:
            pos = torch.arange(S, device=x.device, dtype=torch.float32)
        pos = pos.float().unsqueeze(-1)
        freq = torch.arange(0, C, 2, device=x.device, dtype=torch.float32)
        freq = (freq * (-math.log(self.theta) / C)).exp()
        angle = pos * freq  # [..., S, C // 2]
        embedding = torch.stack([angle.sin(), angle.cos()], dim=-1)
        embedding = embedding.flatten(-2)[..., :C]  # [..., S, C]
        return x + embedding.to(x.dtype)
