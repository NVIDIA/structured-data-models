# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence

import torch
from torch import Tensor

from sdm.prior import Mixture
from sdm.prior.matrix_samplers import MatrixSampler


class MixtureMatrixSampler(MatrixSampler, Mixture[MatrixSampler]):
    """Pick one matrix sampler per call from a weighted pool.

    Args:
        samplers: Matrix samplers or ``(sampler, weight)`` pairs.
        correlated: Share selection probabilities within an active
            :class:`~sdm.prior.CorrelatedContext`.
        key: Key for shared probabilities. Defaults to this mixture's
            identity.
    """

    def __init__(
        self,
        samplers: Sequence[MatrixSampler | tuple[MatrixSampler, float]],
        correlated: bool = False,
        key: Hashable | None = None,
    ) -> None:
        Mixture.__init__(self, samplers, correlated, key)

    def sample(  # noqa: D102
        self,
        size: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.pick(generator=generator).sample(
            size,
            dtype=dtype,
            generator=generator,
        )
