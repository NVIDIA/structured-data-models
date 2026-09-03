# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import Literal

import pytest
import torch

from sdm import NaT, Stype, TableTensor
from sdm.processing import AddCalendarFields


def _timestamp(value: datetime) -> int:
    assert value.tzinfo is UTC
    return int(value.timestamp() * 1_000_000)


@pytest.mark.parametrize("encoding", ["raw", "cyclic"])
def test_add_calendar_fields(encoding: Literal["raw", "cyclic"]) -> None:
    table = TableTensor(
        datetime=torch.tensor(
            [
                [_timestamp(datetime(2024, 2, 29, 23, 59, tzinfo=UTC))],
                [_timestamp(datetime(2023, 3, 1, tzinfo=UTC))],
                [_timestamp(datetime(1969, 12, 31, 23, 1, tzinfo=UTC))],
                [NaT],
            ]
        ),
    )

    encoder = AddCalendarFields(
        fields=["minute", "hour", "weekday", "day_of_month", "month"],
        encoding=encoding,
    )
    output = encoder.transform(table)

    if encoding == "raw":
        assert output.columns[Stype.datetime] == ("dt_0",)
        assert output.columns[Stype.numerical] == (
            "dt_0__minute",
            "dt_0__hour",
            "dt_0__weekday",
            "dt_0__day_of_month",
            "dt_0__month",
        )
        torch.testing.assert_close(
            output.numerical,
            torch.tensor(
                [
                    [59, 23, 3, 28, 1],
                    [0, 0, 2, 0, 2],
                    [1, 23, 2, 30, 11],
                    [float("NaN")] * 5,
                ]
            ),
            equal_nan=True,
        )
    else:
        assert output.columns[Stype.datetime] == ("dt_0",)
        assert output.columns[Stype.numerical] == (
            "dt_0__minute__sin",
            "dt_0__minute__cos",
            "dt_0__hour__sin",
            "dt_0__hour__cos",
            "dt_0__weekday__sin",
            "dt_0__weekday__cos",
            "dt_0__day_of_month__sin",
            "dt_0__day_of_month__cos",
            "dt_0__month__sin",
            "dt_0__month__cos",
        )
