# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior._cdist import cdist_argmin
from sdm.prior.functions import RandomFunction, RandomLinear


class RandomDiscretization(RandomFunction):
    """Replace each row with a linear transformation of its nearest center.

    Args:
        num_centers: Number of centers, capped at the row count.
        p: Minkowski distance exponent.
        linear: Random linear function. Defaults to :class:`RandomLinear`.
    """

    def __init__(
        self,
        num_centers: HP[int] | int,
        p: HP[float] | float,
        linear: RandomLinear | None = None,
    ) -> None:
        self.num_centers = HP.from_value(num_centers)
        self.p = HP.from_value(p)
        self.linear = linear or RandomLinear()

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        num_rows = x.size(0)
        num_centers = min(
            self.num_centers.sample(generator=generator), num_rows
        )

        # Pick distinct center rows without replacement.
        center_index = torch.rand(
            size=(num_rows,), device=x.device, generator=generator
        ).argsort()[:num_centers]
        center = x[center_index]
        p = self.p.sample(generator=generator)
        nearest = cdist_argmin(x, center, p=p)

        center = self.linear.forward(
            x=center,
            out_channels=out_channels,
            generator=generator,
        )
        return center[nearest]
