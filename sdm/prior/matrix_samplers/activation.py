# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.activations import Activation
from sdm.prior.matrix_samplers import MatrixSampler
from sdm.prior.matrix_samplers._normalize import normalize


class ActivationMatrixSampler(MatrixSampler):
    """Sample a base matrix and apply a random activation.

    Activation operates on flattened matrix entries before adding Gaussian
    noise of scale ``1e-3``.

    Args:
        matrix_sampler: Base matrix sampler; use ``normalize=False`` for
            raw inputs.
        activation: Activation applied to flattened matrix entries.
        normalize: Add small Gaussian noise and L2-normalize output rows.
    """

    def __init__(
        self,
        matrix_sampler: MatrixSampler,
        activation: Activation,
        normalize: bool = True,
    ) -> None:
        self.matrix_sampler = matrix_sampler
        self.activation = activation
        self.normalize = normalize

    def sample(  # noqa: D102
        self,
        size: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        W = self.matrix_sampler.sample(
            size,
            dtype=dtype,
            generator=generator,
        )
        W = self.activation.forward(
            W.flatten(-2, -1),
            generator=generator,
        ).view_as(W)
        noise = torch.randn(
            W.size(),
            dtype=W.dtype,
            device="cpu",
            generator=generator,
        )
        W.add_(noise, alpha=1e-3)
        return normalize(W, generator) if self.normalize else W
