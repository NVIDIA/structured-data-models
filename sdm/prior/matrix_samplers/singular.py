# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior.matrix_samplers import MatrixSampler
from sdm.prior.matrix_samplers._normalize import normalize
from sdm.prior.util import random_weight


class SingularValuesMatrixSampler(MatrixSampler):
    """Sample matrices factored as ``U diag(w) V^T``.

    Uses Gaussian ``U``, ``V`` and random power-law weights ``w``.

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
        *batch, m, k = size
        d = min(m, k)
        U = torch.randn(
            size=(*batch, m, d),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        V = torch.randn(
            size=(*batch, d, k),
            dtype=dtype,
            device="cpu",
            generator=generator,
        )
        w = random_weight(
            size=(*batch, d),
            dtype=dtype,
            generator=generator,
        )
        U *= w.unsqueeze(-2)
        W = U @ V
        return normalize(W, generator) if self.normalize else W
