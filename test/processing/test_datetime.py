from datetime import datetime, timezone

import torch
from sdm import Stype, TableTensor
from sdm.processing import EncodeDatetime


def _timestamp(value: datetime) -> int:
    assert value.tzinfo is timezone.utc
    return int(value.timestamp() * 1_000_000)


def test_datetime_features_calendar_channels_and_missing_values() -> None:
    missing = torch.iinfo(torch.int64).min
    utc = timezone.utc
    table = TableTensor(
        columns={Stype.datetime: ("event_time",)},
        datetime=torch.tensor(
            [
                [_timestamp(datetime(2024, 2, 29, 23, 59, tzinfo=utc))],
                [_timestamp(datetime(2023, 3, 1, tzinfo=utc))],
                [_timestamp(datetime(1969, 12, 31, 23, 1, tzinfo=utc))],
                [missing],
            ]
        ),
    )

    output = EncodeDatetime(
        ["minute", "hour", "weekday", "day_of_month", "month"]
    ).transform(table)

    assert output.columns[Stype.datetime] == ("event_time",)
    assert output.columns[Stype.numerical] == (
        "event_time__minute",
        "event_time__hour",
        "event_time__weekday",
        "event_time__day_of_month",
        "event_time__month",
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
