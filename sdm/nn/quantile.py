# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torch import Tensor


class MonotoneQuantiles(torch.nn.Module):
    """Construct ordered quantiles from a median and positive increments.

    Args:
        levels: Increasing probability levels containing ``0.5``.
        width_factor: Multiplier for probability-weighted softplus increments.
    """

    widths: Tensor

    def __init__(
        self, levels: Sequence[float], width_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.median_index = list(levels).index(0.5)
        widths = [
            levels[i + 1] - levels[i]
            if i < self.median_index
            else levels[i] - levels[i - 1]
            if i > self.median_index
            else 0.0
            for i in range(len(levels))
        ]
        self.register_buffer("widths", torch.tensor(widths), persistent=False)
        self.width_factor = width_factor

    def forward(self, x: Tensor) -> Tensor:
        """Return ordered quantiles with the same shape as ``x``.

        Args:
            x: Raw parameters ``[..., Q]``, with ``Q`` probability levels.
        """
        m = self.median_index
        increments = (
            self.width_factor
            * self.widths.to(x)
            * torch.nn.functional.softplus(x)
        )
        median = x[..., m : m + 1]
        up = increments[..., m + 1 :].cumsum(-1)
        down = increments[..., :m].flip(-1).cumsum(-1).flip(-1)
        limit = torch.finfo(x.dtype).max
        return torch.cat([median - down, median, median + up], dim=-1).clamp(
            -limit, limit
        )
