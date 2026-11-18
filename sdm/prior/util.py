# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm.prior import LogUniform


def standardize(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Center and scale values using population standard deviation plus 1e-4.

    Args:
        x: Floating-point tensor to standardize.
        dim: Dimension over which to compute statistics.

    Returns:
        Standardized values and inverse standard deviation, with the
        reduced dimension retained for broadcasting.
    """
    std, mean = torch.std_mean(x, dim=dim, keepdim=True, correction=0)
    std = std + 1e-4
    return (x - mean) / std, 1.0 / std


def random_weight(
    size: torch.Size | tuple[int, ...],
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Sample independent random weight vectors.

        w_i = i^(-q) * exp(sigma * N(0, 1)),   1 <= i <= channels

    with ``q ~ LogUniform(0.1 / log(channels + 1), 6)`` and
    ``sigma ~ LogUniform(1e-4, 10)`` drawn independently per vector. The
    last dim of ``size`` is the channels dim; vectors are scaled to L2 norm
    ``sqrt(channels)`` and independently shuffled along it.

    Args:
        size: Output shape; the last dimension contains each weight vector.
        dtype: Output dtype. Defaults to the default floating-point dtype.
        generator: Pseudorandom generator.
    """
    dtype = dtype or torch.get_default_dtype()
    q = (
        LogUniform(
            low=0.1 / math.log(size[-1] + 1),
            high=6.0,
        )
        .sample_batch(
            size=math.prod(size[:-1]),
            generator=generator,
        )
        .to(dtype)
        .view(*size[:-1], 1)
    )
    sigma = (
        LogUniform(
            low=1e-4,
            high=10.0,
        )
        .sample_batch(
            size=math.prod(size[:-1]),
            generator=generator,
        )
        .to(dtype)
        .view(*size[:-1], 1)
    )

    m = torch.arange(1, size[-1] + 1, dtype=dtype, device="cpu")
    noise = torch.randn(size, dtype=dtype, device="cpu", generator=generator)
    log_w = sigma * noise - q * m.log()
    w = (log_w - log_w.amax(dim=-1, keepdim=True)).exp()
    w = w / w.norm(dim=-1, keepdim=True)

    perm = torch.argsort(
        input=torch.rand(size, device="cpu", generator=generator),
        dim=-1,
    )
    return math.sqrt(size[-1]) * w.gather(-1, perm)
