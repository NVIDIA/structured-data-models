# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torch import Tensor

from sdm.prior import Mixture
from sdm.prior.functions import ParentMerger, RandomFunction


class MixtureFunction(RandomFunction, Mixture[RandomFunction]):
    """Pick one random function per call from a weighted pool.

    Args:
        functions: Functions or ``(function, weight)`` pairs. Bare functions
            have weight one.
        correlated: Share selection probabilities within a
            :class:`~sdm.prior.CorrelatedContext`.
    """

    def __init__(
        self,
        functions: Sequence[RandomFunction | tuple[RandomFunction, float]],
        correlated: bool = True,
    ) -> None:
        Mixture.__init__(self, items=functions, correlated=correlated)

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.pick(generator=generator).forward(
            x=x,
            out_channels=out_channels,
            generator=generator,
        )


class MixtureParentMerger(ParentMerger, Mixture[ParentMerger]):
    """Pick one parent merger per call from a weighted pool.

    Args:
        mergers: Parent mergers or ``(merger, weight)`` pairs. Bare mergers
            have weight one.
        correlated: Share selection probabilities within a
            :class:`~sdm.prior.CorrelatedContext`.
    """

    def __init__(
        self,
        mergers: Sequence[ParentMerger | tuple[ParentMerger, float]],
        correlated: bool = True,
    ) -> None:
        Mixture.__init__(self, items=mergers, correlated=correlated)

    def forward(  # noqa: D102
        self,
        parent_values: Sequence[Tensor],
        random_fn: RandomFunction,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.pick(generator=generator).forward(
            parent_values=parent_values,
            random_fn=random_fn,
            out_channels=out_channels,
            generator=generator,
        )
