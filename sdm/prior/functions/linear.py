# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.functions import RandomFunction
from sdm.prior.matrix_samplers import (
    GaussianMatrixSampler,
    MatrixSampler,
)


class RandomLinear(RandomFunction):
    """Apply a random linear function.

    Args:
        matrix_sampler: Random matrix sampler. Defaults to
            :class:`~sdm.prior.GaussianMatrixSampler`.
    """

    def __init__(self, matrix_sampler: MatrixSampler | None = None) -> None:
        self.matrix_sampler = matrix_sampler or GaussianMatrixSampler()

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        _, in_channels = x.size()
        matrix = self.matrix_sampler.sample(
            size=(1, out_channels, in_channels),
            dtype=x.dtype,
            generator=generator,
        ).transpose(-2, -1)
        # Preserve the reference accumulation order with a singleton batch.
        return torch.bmm(x.unsqueeze(0), matrix).squeeze(0)
