# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior.root_samplers import RootSampler


class UniformRootSampler(RootSampler):
    """Sample root values from a uniform distribution.

    Args:
        low: Lower bound.
        high: Upper bound.
    """

    def __init__(
        self,
        low: HP[float] | float = -1.0,
        high: HP[float] | float = 1.0,
    ) -> None:
        self.low = HP.from_value(low)
        self.high = HP.from_value(high)

    def sample(  # noqa: D102
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        low = self.low.sample(generator=generator)
        high = self.high.sample(generator=generator)
        return low + torch.rand(
            size=(num_rows, channels),
            device="cpu",
            generator=generator,
        ) * (high - low)
