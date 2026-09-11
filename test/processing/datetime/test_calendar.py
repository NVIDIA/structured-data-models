from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NaT,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import AddCalendarFields
from sdm.testing import withCUDA


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


@withCUDA
@pytest.mark.parametrize("encoding", ["raw", "cyclic"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_calendar_fields_across_timestamp_range(
    device: torch.device,
    encoding: Literal["raw", "cyclic"],
    dtype: torch.dtype,
) -> None:
    timestamps = np.array(
        [
            np.iinfo(np.int64).min + 1,
            _timestamp(datetime(1900, 2, 28, 23, 59, tzinfo=UTC)),
            -1,
            0,
            _timestamp(datetime(2000, 2, 29, 12, 34, tzinfo=UTC)),
            np.iinfo(np.int64).max,
        ],
        dtype=np.int64,
    )
    # Avoid NumPy unit-conversion overflow near the int64 timestamp limits.
    days = (timestamps // 86_400_000_000).astype("datetime64[D]")
    months = days.astype("datetime64[M]")
    raw = np.stack(
        [
            (timestamps // 60_000_000) % 60,
            (timestamps // 3_600_000_000) % 24,
            (days.astype(np.int64) + 3) % 7,
            (days - months.astype("datetime64[D]")).astype(np.int64),
            months.astype(np.int64) % 12,
        ],
        axis=-1,
    )
    expected = torch.tensor(raw, device=device)
    if encoding == "cyclic":
        month_days = (
            (months + 1).astype("datetime64[D]")
            - months.astype("datetime64[D]")
        ).astype(np.int64)
        normalized = torch.stack(
            [
                expected[:, 0] / 60.0,
                expected[:, 1] / 24.0,
                expected[:, 2] / 7.0,
                expected[:, 3] / torch.tensor(month_days, device=device),
                expected[:, 4] / 12.0,
            ],
            dim=-1,
        )
        angles = normalized.to(dtype) * (2 * torch.pi)
        expected = torch.stack([angles.sin(), angles.cos()], dim=-1)
        expected = expected.flatten(-2, -1)
    expected = expected.to(dtype)
    expected = torch.cat(
        [expected, expected.new_full((1, expected.size(-1)), float("nan"))],
        dim=0,
    )
    values = torch.tensor([*timestamps, NaT], device=device)
    # Batched, expanded, noncontiguous input with two datetime columns.
    data = torch.stack([values, values.flip(0)]).expand(2, -1, -1)
    data = data.transpose(-2, -1)
    numerical = torch.full(
        (*data.shape[:-1], 1), 7, device=device, dtype=dtype
    )
    table = TableTensor(datetime=data, numerical=numerical)
    original = data.clone()
    processor = AddCalendarFields(
        fields=["minute", "hour", "weekday", "day_of_month", "month"],
        encoding=encoding,
    )

    output = processor.transform(table)

    expected = torch.stack([expected, expected.flip(0)], dim=-2)
    expected = expected.flatten(-2, -1).expand(2, -1, -1)
    torch.testing.assert_close(
        output.numerical[..., 1:], expected, equal_nan=True
    )
    torch.testing.assert_close(output.numerical[..., :1], numerical)
    torch.testing.assert_close(output.datetime, original)
    torch.testing.assert_close(table.datetime, original)


@withCUDA
@pytest.mark.parametrize("encoding", ["raw", "cyclic"])
@pytest.mark.parametrize(
    "field", ["minute", "hour", "weekday", "day_of_month", "month"]
)
def test_single_calendar_field_matches_combined(
    device: torch.device,
    encoding: Literal["raw", "cyclic"],
    field: Literal["minute", "hour", "weekday", "day_of_month", "month"],
) -> None:
    table = TableTensor(
        datetime=torch.tensor(
            [
                [torch.iinfo(torch.int64).min + 1],
                [-1],
                [_timestamp(datetime(2024, 2, 29, tzinfo=UTC))],
                [torch.iinfo(torch.int64).max],
                [NaT],
            ],
            device=device,
        )
    )
    combined = AddCalendarFields(
        fields=["month", "weekday", "minute", "day_of_month", "hour"],
        encoding=encoding,
    ).transform(table)
    output = AddCalendarFields(fields=[field], encoding=encoding).transform(
        table
    )

    expected = combined[list(output.columns[Stype.numerical])].numerical
    torch.testing.assert_close(output.numerical, expected, equal_nan=True)


@pytest.mark.parametrize("encoding", ["raw", "cyclic"])
def test_calendar_fields_preserve_mixed_blocks_and_numerical_prefix(
    encoding: Literal["raw", "cyclic"],
) -> None:
    numerical = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]
    timestamps = (
        torch.tensor([[0, -1], [NaT, 86_400_000_000], [1, -86_400_000_000]])
        .T.contiguous()
        .T
    )
    table = TableTensor(
        columns={
            Stype.numerical: ("height", "mass"),
            Stype.datetime: ("visit", "birth"),
        },
        numerical=numerical,
        datetime=timestamps,
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [-1]], dtype=torch.int32),
            categories=(torch.arange(2),),
        ),
        text=StringTensor.from_list([["first"], ["second"], [None]]),
        id=ColumnarTensor((torch.arange(3),)),
    )
    original = table.clone()
    processor = AddCalendarFields(
        fields=["month", "minute", "weekday"],
        encoding=encoding,
    )
    datetime_only = TableTensor(
        columns={Stype.datetime: table.columns[Stype.datetime]},
        datetime=timestamps,
        numerical=numerical[..., :0],
    )
    expected = processor.transform(datetime_only)

    output = processor.transform(table)

    assert output.columns[Stype.numerical] == (
        *table.columns[Stype.numerical],
        *expected.columns[Stype.numerical],
    )
    torch.testing.assert_close(output.numerical[..., :2], numerical)
    torch.testing.assert_close(
        output.numerical[..., 2:], expected.numerical, equal_nan=True
    )
    for stype in (Stype.categorical, Stype.datetime, Stype.text, Stype.id):
        assert output.columns[stype] == table.columns[stype]
        assert output.blocks[stype] is table.blocks[stype]
    output.numerical[..., :2].zero_()
    assert table.equal(original)
