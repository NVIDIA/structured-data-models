# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior.functions import ParentMerger, RandomFunction


class ReduceParentMerger(ParentMerger):
    """Apply ``random_fn`` to each parent and reduce the outputs.

    Args:
        aggr: Reduction over parents, such as ``sum``, ``prod``, ``amax``,
            or ``logsumexp``.
    """

    def __init__(
        self,
        aggr: HP[str] | str,
    ) -> None:
        self.aggr = HP.from_value(aggr)

    def forward(  # noqa: D102
        self,
        parent_values: Sequence[Tensor],
        random_fn: RandomFunction,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        x = torch.stack(
            [
                random_fn.forward(
                    x=x,
                    out_channels=out_channels,
                    generator=generator,
                )
                for x in parent_values
            ],
            dim=0,
        )
        aggr = self.aggr.sample(generator=generator)
        return getattr(torch, aggr)(x, dim=0)
