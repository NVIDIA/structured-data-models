# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.functions import RandomFunction
from sdm.prior.matrix_samplers import (
    GaussianMatrixSampler,
    MatrixSampler,
)


class RandomQuadratic(RandomFunction):
    """Apply a random quadratic function.

    Args:
        max_channels: Maximum number of input channels used in the quadratic
            form.
        matrix_sampler: Random matrix sampler. Defaults to
            :class:`~sdm.prior.GaussianMatrixSampler`.
    """

    def __init__(
        self,
        max_channels: int = 20,
        matrix_sampler: MatrixSampler | None = None,
    ) -> None:
        self.max_channels = max_channels
        self.matrix_sampler = matrix_sampler or GaussianMatrixSampler()

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        num_rows, in_channels = x.size()

        # Subsample input channels to keep the quadratic form small:
        perm = torch.randperm(
            n=in_channels,
            device=x.device,
            generator=generator,
        )[: self.max_channels]
        x = x[..., perm]

        # Append a bias channel:
        ones = x.new_ones((num_rows, 1))
        x = torch.cat([x, ones], dim=-1)

        # Output matrices share one dataset's kernel geometry.
        matrix = self.matrix_sampler.sample(
            size=(1, out_channels, x.size(-1), x.size(-1)),
            dtype=x.dtype,
            generator=generator,
        ).squeeze(0)

        return torch.einsum("rj,ijk,rk->ri", x, matrix, x)
