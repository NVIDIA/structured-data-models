# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def uniform(
    low: float,
    high: float,
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.empty(size, device=device, dtype=dtype).uniform_(
        low, high, generator=generator
    )


def log_uniform(
    low: float,
    high: float,
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    return uniform(
        low=math.log(low),
        high=math.log(high),
        size=size,
        device=device,
        dtype=dtype,
        generator=generator,
    ).exp()
