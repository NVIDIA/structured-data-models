# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

from torch import Tensor

from sdm.tensor.datetime import DatetimeUnit, convert_datetime, datetime_parts


class TemporalClock:
    """Normalize sampling clocks without changing stored feature timestamps.

    All-microsecond graphs retain their full range and raw clock values.
    Graphs with nanosecond event timestamps use nanoseconds; mixed clocks
    must fit the int64 nanosecond timestamp range.
    """

    def __init__(
        self, times: Mapping[str, tuple[Tensor, DatetimeUnit]]
    ) -> None:
        self.unit: DatetimeUnit = (
            "ns" if any(unit == "ns" for _, unit in times.values()) else "us"
        )
        self.times = {
            name: convert_datetime(value, unit, self.unit)
            for name, (value, unit) in times.items()
        }

    def query_time(self, value: Tensor, unit: DatetimeUnit) -> Tensor:
        if unit == "ns" and self.unit == "us":
            # For an inclusive comparison against microsecond event clocks,
            # t_us <= t_ns is exactly t_us <= floor(t_ns / 1000), including
            # negative epochs. Only this comparison bound is derived; task
            # tensors and their calendar/relative-time features stay lossless.
            return datetime_parts(value, unit)[0]
        return convert_datetime(value, unit, self.unit)
