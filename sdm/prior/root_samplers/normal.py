# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior.root_samplers import RootSampler


class NormalRootSampler(RootSampler):
    """Sample root values from a normal distribution.

    Args:
        mean: Normal mean.
        std: Normal standard deviation.
    """

    def __init__(
        self,
        mean: HP[float] | float = 0.0,
        std: HP[float] | float = 1.0,
    ) -> None:
        self.mean = HP.from_value(mean)
        self.std = HP.from_value(std)

    def sample(  # noqa: D102
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        mean = self.mean.sample(generator=generator)
        std = self.std.sample(generator=generator)
        rnd = torch.randn(
            size=(num_rows, channels),
            device="cpu",
            generator=generator,
        )
        return mean + rnd * std
