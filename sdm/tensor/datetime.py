# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from sdm import NaT

DatetimeUnit = Literal["us", "ns"]
US_PER_DAY = 24 * 60 * 60 * 1_000_000
NS_PER_DAY = US_PER_DAY * 1_000


def datetime_parts(value: Tensor, unit: DatetimeUnit) -> tuple[Tensor, Tensor]:
    """Return exact microseconds and a nanosecond remainder for ordering."""
    if unit == "us":
        return value, torch.zeros_like(value)
    coarse = value.div(1_000, rounding_mode="floor")
    coarse = coarse.masked_fill(value == NaT, NaT)
    return coarse, value.remainder(1_000).masked_fill(value == NaT, 0)


def convert_datetime(
    value: Tensor,
    source: DatetimeUnit,
    target: DatetimeUnit,
) -> Tensor:
    """Convert a datetime block without dropping precision or overflowing.

    Cross-unit conversion validates user data at the conversion boundary,
    synchronizing CUDA inputs so invalid data raises without a device assert.
    Equal-unit operations retain the original tensor without validation.
    """
    if source == target:
        return value
    valid = value != NaT
    if source == "us":
        low = (torch.iinfo(torch.int64).min + 1 + 999) // 1_000
        high = torch.iinfo(torch.int64).max // 1_000
        if not ((~valid) | ((value >= low) & (value <= high))).all():
            raise ValueError(
                "Datetime value is outside the nanosecond timestamp range"
            )
        return (value.masked_fill(~valid, 0) * 1_000).masked_fill(~valid, NaT)
    if not ((~valid) | (value.remainder(1_000) == 0)).all():
        raise ValueError("Datetime conversion would lose nanosecond precision")
    return value.div(1_000, rounding_mode="floor").masked_fill(~valid, NaT)


def units_per_day(
    units: Sequence[DatetimeUnit], device: torch.device
) -> int | Tensor:
    """Return scalar or column-wise clock ticks per calendar day."""
    if all(unit == "us" for unit in units):
        return US_PER_DAY
    if all(unit == "ns" for unit in units):
        return NS_PER_DAY
    return torch.tensor(
        [US_PER_DAY if unit == "us" else NS_PER_DAY for unit in units],
        dtype=torch.int64,
        device=device,
    )


def relative_days(
    value: Tensor,
    units: Sequence[DatetimeUnit],
    seed: Tensor,
    seed_units: Sequence[DatetimeUnit],
) -> Tensor:
    """Compute relative day features while preserving subsecond differences."""
    if all(unit == "us" for unit in (*units, *seed_units)):
        return (seed.unsqueeze(-2) - value.unsqueeze(-1)) / US_PER_DAY
    value_scale = units_per_day(units, value.device) // (24 * 60 * 60)
    seed_scale = units_per_day(seed_units, seed.device) // (24 * 60 * 60)
    # Split before subtracting: absolute epoch nanoseconds lose precision in
    # float64, and a direct int64 subtraction can overflow.
    seconds = seed.div(seed_scale, rounding_mode="floor").unsqueeze(-2)
    seconds = seconds - value.div(
        value_scale, rounding_mode="floor"
    ).unsqueeze(-1)
    seed_ns = seed.remainder(seed_scale) * (1_000_000_000 // seed_scale)
    value_ns = value.remainder(value_scale) * (1_000_000_000 // value_scale)
    nanos = seed_ns.unsqueeze(-2) - value_ns.unsqueeze(-1)
    seconds = seconds + nanos.div(1_000_000_000, rounding_mode="floor")
    nanos = nanos.remainder(1_000_000_000)
    # Match signs to preserve +/-1ns across a second boundary.
    borrow = (seconds < 0) & (nanos != 0)
    seconds = seconds + borrow
    nanos = nanos - borrow * 1_000_000_000
    result = (
        seconds.to(torch.float64) + nanos.to(torch.float64) / 1_000_000_000
    ) / (24 * 60 * 60)
    return result.to(torch.get_default_dtype())
