from datetime import datetime, timezone

import torch
from sdm import Stype, TableTensor
from sdm.processing import DatetimeFeatures
from sdm.testing import withCUDA


def _timestamp(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000)


def test_datetime_features_calendar_channels_and_missing_values() -> None:
    missing = torch.iinfo(torch.int64).min
    table = TableTensor(
        columns={Stype.datetime: ("event_time",)},
        datetime=torch.tensor(
            [
                [
                    _timestamp(
                        datetime(2024, 2, 29, 23, 59, tzinfo=timezone.utc)
                    )
                ],
                [_timestamp(datetime(2023, 3, 1, tzinfo=timezone.utc))],
                [missing],
            ]
        ),
    )

    output = DatetimeFeatures().transform(table)

    assert output.columns[Stype.numerical] == (
        "event_time__minute",
        "event_time__hour",
        "event_time__weekday",
        "event_time__day_of_month",
        "event_time__day_of_year",
    )
    assert output.columns[Stype.datetime] == ()
    torch.testing.assert_close(
        output.numerical,
        torch.tensor(
            [
                [59 / 60, 23 / 24, 3 / 7, 28 / 29, 59 / 366],
                [0, 0, 2 / 7, 0, 59 / 365],
                [0, 0, 0, 0, 0],
            ]
        ),
    )


@withCUDA
def test_datetime_features_preserves_device_dtype_and_raw_datetime(
    device: torch.device,
) -> None:
    timestamp = torch.tensor(
        [[0, 60 * 1_000_000]],
        dtype=torch.int64,
        device=device,
    )
    table = TableTensor(
        columns={Stype.datetime: ("created_at", "updated_at")},
        numerical=torch.empty((1, 0), dtype=torch.float64, device=device),
        datetime=timestamp,
    )

    output = DatetimeFeatures(preserve_datetime=True).transform(table)

    assert output.numerical.shape == (1, 10)
    assert output.numerical.dtype == torch.float64
    assert output.device == timestamp.device
    assert output.columns[Stype.datetime] == ("created_at", "updated_at")
    assert output.datetime is timestamp


def test_tabiclv2_recipe_encodes_and_preserves_datetime() -> None:
    from sdm.models.tabiclv2.recipe import default_recipe

    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.datetime: ("event_time",),
        },
        numerical=torch.tensor([[1.0], [2.0], [3.0]]),
        datetime=torch.tensor(
            [
                [_timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))],
                [_timestamp(datetime(2024, 1, 2, tzinfo=timezone.utc))],
                [_timestamp(datetime(2024, 1, 3, tzinfo=timezone.utc))],
            ]
        ),
    )

    output = default_recipe().features.fit_transform(table)

    assert output.columns[Stype.datetime] == ("event_time",)
    assert torch.equal(output.datetime, table.datetime)
    assert "event_time__weekday" in output.columns[Stype.numerical]
    assert "event_time__day_of_year" in output.columns[Stype.numerical]
