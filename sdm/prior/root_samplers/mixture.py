# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence

import torch
from torch import Tensor

from sdm.prior import Mixture
from sdm.prior.root_samplers import RootSampler


class MixtureRootSampler(RootSampler, Mixture[RootSampler]):
    r"""Pick one root sampler per call from a weighted pool.

    Args:
        samplers: Root samplers or ``(sampler, weight)`` pairs.
        correlated: Share selection probabilities within an active
            :class:`~sdm.prior.CorrelatedContext`.
        key: Key for shared probabilities. Defaults to this mixture's
            identity.
    """

    def __init__(
        self,
        samplers: Sequence[RootSampler | tuple[RootSampler, float]],
        correlated: bool = True,
        key: Hashable | None = None,
    ) -> None:
        Mixture.__init__(self, samplers, correlated, key)

    def sample(  # noqa: D102
        self,
        num_rows: int,
        channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        sampler = self.pick(generator=generator)
        return sampler.sample(
            num_rows=num_rows,
            channels=channels,
            generator=generator,
        )
