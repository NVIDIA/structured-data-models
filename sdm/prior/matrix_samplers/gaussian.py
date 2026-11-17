# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.matrix_samplers import MatrixSampler
from sdm.prior.matrix_samplers._normalize import normalize


class GaussianMatrixSampler(MatrixSampler):
    """Sample random Gaussian matrices.

    Args:
        normalize: Add small Gaussian noise and L2-normalize rows.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def sample(  # noqa: D102
        self,
        size: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        W = torch.randn(
            size=size,
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        return normalize(W, generator) if self.normalize else W
