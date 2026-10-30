# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from datetime import datetime
from typing import cast

import pandas as pd
import pyarrow as pa
import pytest
import torch

from sdm import EnsembleTable, NaT, TableTensor
from sdm.processing import AddCalendarFields
from sdm.tensor.datetime import relative_days
from sdm.testing import onlyCUDA, withCUDA


def test_lossless_datetime_ingress_and_roundtrip() -> None:
    arrow = pa.table(
        {
            "wide": pa.array(
                [datetime(3000, 1, 1), None], type=pa.timestamp("us")
            ),
            "precise": pa.array([1, -1], type=pa.timestamp("ns")),
            "aligned": pa.array([1_000, -1_000], type=pa.timestamp("ns")),
        }
    )
    table = TableTensor.from_arrow(
        arrow, dict.fromkeys(arrow.column_names, "datetime")
    )
    assert table.datetime_units == ("us", "ns", "us")
    result = table.to_arrow()
    assert result["wide"].equals(arrow["wide"])
    assert result["precise"].equals(arrow["precise"])
    assert result["aligned"].cast(pa.timestamp("ns")).equals(arrow["aligned"])
    frame = pd.DataFrame({"time": pd.to_datetime([1, -1, None], unit="ns")})
    pandas_table = TableTensor.from_pandas(frame, {"time": "datetime"})
    assert pandas_table.datetime_units == ("ns",)
    pd.testing.assert_frame_equal(pandas_table.to_pandas(), frame)


@withCUDA
def test_precision_metadata_survives_table_operations_and_pickle(
    device: torch.device,
) -> None:
    table = TableTensor.from_arrow(
        pa.table({"time": pa.array([1, -1, None], type=pa.timestamp("ns"))}),
        {"time": "datetime"},
        device=device,
    )
    copies = [
        table.clone(),
        table.to(dtype=torch.float64),
        table[1:],
        table.unsqueeze(0).expand(2, -1, -1)[0],
        table["time"],
        table.select_stypes("datetime"),
        table.replace_blocks(datetime=table.datetime.clone()),
        pickle.loads(pickle.dumps(table)),
        torch.cat(
            [table, TableTensor(numerical=torch.ones(3, 1, device=device))],
            dim=-1,
        ),
    ]
    for copy in copies:
        copy = cast(TableTensor, copy)
        assert copy.datetime_units == ("ns",)
        expected = table[1:] if len(copy) == 2 else table
        assert copy["time"].to_arrow().equals(expected.to_arrow())
    assert table.drop_stypes("datetime").datetime_units == ()


def test_mixed_precision_rows_promote_losslessly() -> None:
    micros = TableTensor(datetime=torch.tensor([[1], [-1], [NaT]]))
    nanos = TableTensor(
        datetime=torch.tensor([[1_000], [-1_000], [NaT]]),
        datetime_units=("ns",),
    )
    assert micros.equal(nanos)
    assert micros.allclose(nanos)
    for result in [torch.cat([micros, nanos]), torch.stack([micros, nanos])]:
        result = cast(TableTensor, result)
        assert result.datetime_units == ("ns",)
        assert result.datetime.flatten().tolist() == [1_000, -1_000, NaT] * 2
    named_nanos = TableTensor(
        columns={"datetime": ("ns",)},
        datetime=nanos.datetime,
        datetime_units=nanos.datetime_units,
    )
    mixed_columns = cast(TableTensor, torch.cat([micros, named_nanos], dim=-1))
    assert mixed_columns.datetime_units == ("us", "ns")
    assert mixed_columns["ns"].datetime_units == ("ns",)


def test_heterogeneous_ensemble_preserves_precision_and_wide_dates() -> None:
    wide = TableTensor(datetime=torch.tensor([[32_503_680_000_000_000]]))
    precise = TableTensor(datetime=torch.tensor([[1]]), datetime_units=("ns",))
    tables = [wide, precise]
    ensemble = EnsembleTable.from_tables(tables, member_table_ids=[0, 1])
    packed = ensemble.replace_tables(tables=tables, member_table_ids=[0, 1])
    for output in [ensemble, packed]:
        for i, expected in enumerate(tables):
            assert output[i].datetime_units == expected.datetime_units
            assert output[i].to_arrow().equals(expected.to_arrow())


def test_new_mixed_precision_range_error_does_not_change_legacy_data() -> None:
    wide = TableTensor(datetime=torch.tensor([[32_503_680_000_000_000]]))
    assert wide.datetime_units == ("us",)
    assert wide.to_arrow()["dt_0"][0].as_py() == datetime(3000, 1, 1)
    precise = TableTensor(datetime=torch.tensor([[1]]), datetime_units=("ns",))
    with pytest.raises(ValueError, match="nanosecond timestamp range"):
        torch.cat([wide, precise])
    assert wide.to_arrow()["dt_0"][0].as_py() == datetime(3000, 1, 1)


def test_mixed_precision_inplace_view_assignment_is_rejected() -> None:
    parent = TableTensor(datetime=torch.tensor([[1], [2]]))
    before = parent.to_arrow()
    view = parent[:1]
    source = TableTensor(
        datetime=torch.tensor([[1_001]]), datetime_units=("ns",)
    )
    with pytest.raises(ValueError, match="matching datetime units"):
        view.copy_(source)
    with pytest.raises(ValueError, match="matching datetime units"):
        parent[:1] = source
    assert parent.to_arrow().equals(before)


def test_legacy_serialized_constructor_defaults_to_microseconds() -> None:
    original = TableTensor(datetime=torch.tensor([[1_700_000_000_000_000]]))

    class LegacyPayload:
        def __reduce__(self):
            # Constructor arguments emitted by the old pickle format.
            return TableTensor, (
                original.size()[:-1],
                original.columns,
                original.numerical,
                original.categorical,
                original.datetime,
                original.text,
                original.id,
            )

    restored = pickle.loads(pickle.dumps(LegacyPayload()))
    assert restored.datetime_units == ("us",)
    assert restored.to_arrow().equals(original.to_arrow())


@withCUDA
def test_nanosecond_calendar_rollover_with_wide_microsecond_column(
    device: torch.device,
) -> None:
    midnight = 1_704_067_200_000_000_000  # 2024-01-01, a Monday.
    arrow = pa.table(
        {
            "legacy": pa.array(
                [datetime(3000, 1, 1)] * 4, type=pa.timestamp("us")
            ),
            "precise": pa.array(
                [midnight - 1, midnight, midnight + 1, None],
                type=pa.timestamp("ns"),
            ),
        }
    )
    table = TableTensor.from_arrow(
        arrow, dict.fromkeys(arrow.column_names, "datetime"), device=device
    )
    output = AddCalendarFields(
        ["minute", "hour", "weekday", "day_of_month", "month"]
    ).transform(table)
    torch.testing.assert_close(
        output.numerical[:, 5:],
        torch.tensor(
            [
                [59, 23, 6, 30, 11],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [float("nan")] * 5,
            ],
            device=device,
        ),
        equal_nan=True,
    )
    assert output.datetime_units == ("us", "ns")
    assert output.select_stypes("datetime").to_arrow().equals(arrow)
    assert output.numerical[0, :5].tolist() == [
        0,
        0,
        datetime(3000, 1, 1).weekday(),
        0,
        0,
    ]


@pytest.mark.parametrize("base", [0, 1_704_067_200_000_000_000])
def test_relative_day_features_preserve_nanosecond_sign_and_legacy_arithmetic(
    base: int,
) -> None:
    value = torch.tensor([[base - 1, base + 1]])
    seed = torch.tensor([[base // 1_000]])
    actual = relative_days(value, ("ns", "ns"), seed, ("us",))
    torch.testing.assert_close(
        actual,
        torch.tensor([[[1 / 86_400_000_000_000], [-1 / 86_400_000_000_000]]]),
        rtol=1e-6,
        atol=0,
    )
    legacy = torch.tensor([[1_700_000_000_000_000]])
    later = legacy + 86_400_000_000
    assert relative_days(legacy, ("us",), later, ("us",)).equal(
        (later.unsqueeze(-2) - legacy.unsqueeze(-1)) / 86_400_000_000
    )


@onlyCUDA
def test_cudf_datetime_precision_roundtrip() -> None:
    cudf = pytest.importorskip("cudf")
    frame = cudf.DataFrame(
        {
            "precise": cudf.Series([1, -1, None], dtype="datetime64[ns]"),
            "aligned": cudf.Series(
                [1_000, -1_000, None], dtype="datetime64[ns]"
            ),
        }
    )
    table = TableTensor.from_cudf(
        frame, {"precise": "datetime", "aligned": "datetime"}
    )
    assert table.datetime_units == ("ns", "us")
    result = table.to_cudf()
    assert result["precise"].equals(frame["precise"])
    assert result["aligned"].astype("datetime64[ns]").equals(frame["aligned"])
    # Existing string-to-datetime ingress remains supported.
    dates = cudf.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
    converted = TableTensor.from_cudf(dates, {"date": "datetime"})
    assert converted.datetime_units == ("us",)


@onlyCUDA
def test_rejected_cuda_conversion_keeps_device_usable() -> None:
    wide = TableTensor(
        datetime=torch.tensor([[32_503_680_000_000_000]], device="cuda")
    )
    precise = TableTensor(
        datetime=torch.tensor([[1]], device="cuda"), datetime_units=("ns",)
    )
    with pytest.raises(ValueError, match="nanosecond timestamp range"):
        torch.cat([wide, precise])
    legacy = TableTensor(datetime=torch.tensor([[2]], device="cuda"))
    combined = cast(TableTensor, torch.cat([legacy, precise]))
    assert combined.datetime_units == ("ns",)
    assert combined.datetime.flatten().tolist() == [2_000, 1]
    assert (torch.ones(1, device="cuda") + 1).item() == 2


def test_compiled_same_unit_table_operation_preserves_precision(
    compilation: None,
) -> None:
    table = TableTensor(
        datetime=torch.tensor([[-1], [1]]), datetime_units=("ns",)
    )
    operation = torch.compile(
        lambda table: table.unsqueeze(0),
        backend="eager",
        fullgraph=True,
    )
    output = operation(table)
    assert output.datetime_units == ("ns",)
    assert output[0].to_arrow().equals(table.to_arrow())
