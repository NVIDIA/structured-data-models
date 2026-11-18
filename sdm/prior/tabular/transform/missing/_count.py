# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch


def _sample_count(
    total: int,
    low: float,
    high: float,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    lower = max(1, math.ceil(total * low))
    upper = max(lower, min(total, math.floor(total * high)))
    return int(
        torch.randint(
            low=lower,
            high=upper + 1,
            size=(),
            device=device,
            generator=generator,
        ).item()
    )
