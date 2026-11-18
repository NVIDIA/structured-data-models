# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence

import torch
from torch import Tensor

from sdm.prior import Mixture
from sdm.prior.activations import Activation


class MixtureActivation(Activation, Mixture[Activation]):
    """Pick one activation per call from a weighted pool.

    Args:
        activations: Activations or ``(activation, weight)`` pairs.
        correlated: Share selection probabilities within an active
            :class:`~sdm.prior.CorrelatedContext`.
        key: Key for shared probabilities. Defaults to this mixture's
            identity.
    """

    def __init__(
        self,
        activations: Sequence[Activation | tuple[Activation, float]],
        correlated: bool = True,
        key: Hashable | None = None,
    ) -> None:
        Mixture.__init__(self, activations, correlated, key)

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return self.pick(generator=generator).forward(x, generator=generator)
