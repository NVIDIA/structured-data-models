from datetime import UTC, datetime

import torch

from sdm import NaT, Stype, TableTensor
from sdm.processing import AddCalendarFields


def _timestamp(value: datetime) -> int:
    assert value.tzinfo is UTC
    return int(value.timestamp() * 1_000_000)


def test_add_calendar_fields_channels_and_missing_values() -> None:
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

    output = AddCalendarFields(
        fields=["minute", "hour", "weekday", "day_of_month", "month"]
    ).transform(table)

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
