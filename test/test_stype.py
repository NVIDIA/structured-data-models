from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest
import torch
from sdm import Stype, infer_stypes


def test_from_pandas() -> None:
    df = pd.DataFrame(
        {
            "age": pd.Series([1, 2], dtype="int64"),
            "income": pd.Series([1.0, 2.5], dtype="float64"),
            "name": pd.Series(["a", "b"], dtype="string"),
            "city": pd.Series(["NY", None], dtype="object"),
            "segment": pd.Series(["x", "y"], dtype="category"),
            "active": pd.Series([True, False], dtype="bool"),
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )

    assert infer_stypes(df) == {
        "age": Stype.numerical,
        "income": Stype.numerical,
        "name": Stype.categorical,
        "city": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
        "created_at": Stype.datetime,
    }


def test_from_arrow() -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "amount": pa.array([Decimal("1.25"), None]),
            "ratio": pa.array([1.0, 2.5], type=pa.float32()),
            "name": pa.array(["a", "b"], type=pa.string()),
            "note": pa.array(["a", "b"], type=pa.large_string()),
            "active": pa.array([True, False], type=pa.bool_()),
            "code": pa.array(["x", "y"]).dictionary_encode(),
            "created_at": pa.array([0, 1], type=pa.timestamp("s")),
        }
    )

    assert infer_stypes(table) == {
        "id": Stype.id,
        "amount": Stype.numerical,
        "ratio": Stype.numerical,
        "name": Stype.categorical,
        "note": Stype.categorical,
        "active": Stype.categorical,
        "code": Stype.categorical,
        "created_at": Stype.datetime,
    }


def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    df = cudf.DataFrame(
        {
            "id": cudf.Series([1, 2], dtype="int64"),
            "age": cudf.Series([25, 31], dtype="int32"),
            "income": cudf.Series([1.0, 2.5], dtype="float64"),
            "amount": cudf.Series(
                [Decimal("1.25"), Decimal("2.50")],
                dtype=cudf.Decimal64Dtype(8, 2),
            ),
            "name": cudf.Series(["a", "b"]),
            "segment": cudf.Series(["x", "y"], dtype="category"),
            "active": cudf.Series([True, False], dtype="bool"),
            "created_at": cudf.Series(
                ["2026-01-01", "2026-01-02"],
                dtype="datetime64[ns]",
            ),
        }
    )

    assert infer_stypes(df) == {
        "id": Stype.id,
        "age": Stype.numerical,
        "income": Stype.numerical,
        "amount": Stype.numerical,
        "name": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
        "created_at": Stype.datetime,
    }


def test_consistent_inference_across_backends() -> None:
    cudf = pytest.importorskip("cudf")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    df = pd.DataFrame(
        {
            "user_id": pd.Series([1, 2], dtype="int64"),
            "age": pd.Series([25, 31], dtype="int32"),
            "income": pd.Series([1.0, 2.5], dtype="float64"),
            "name": pd.Series(["a", "b"], dtype="string"),
            "city": pd.Series(["NY", None], dtype="object"),
            "segment": pd.Series(["x", "y"], dtype="category"),
            "active": pd.Series([True, False], dtype="bool"),
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )
    expected = {
        "user_id": Stype.id,
        "age": Stype.numerical,
        "income": Stype.numerical,
        "name": Stype.categorical,
        "city": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
        "created_at": Stype.datetime,
    }

    assert infer_stypes(df) == expected
    assert (
        infer_stypes(pa.Table.from_pandas(df, preserve_index=False))
        == expected
    )
    assert infer_stypes(cudf.from_pandas(df)) == expected


def test_id_detection() -> None:
    table = pa.table(
        {
            "user_id": pa.array([1, 2], type=pa.int64()),
            "userId": pa.array(["a", "b"], type=pa.string()),
            "order_id_hash": pa.array([1, 2], type=pa.int32()),
            "is_valid": pa.array([True, False], type=pa.bool_()),
            "solid": pa.array([1, 2], type=pa.int64()),
            "covid_cases": pa.array([1, 2], type=pa.int64()),
        }
    )

    assert infer_stypes(table) == {
        "user_id": Stype.id,
        "userId": Stype.id,
        "order_id_hash": Stype.id,
        "is_valid": Stype.categorical,
        "solid": Stype.numerical,
        "covid_cases": Stype.numerical,
    }


def test_overrides() -> None:
    table = pa.table(
        {
            "age": pa.array([25, 31], type=pa.int64()),
            "account_number": pa.array([1, 2], type=pa.int64()),
        }
    )

    assert infer_stypes(table, overrides={"account_number": "id"}) == {
        "age": Stype.numerical,
        "account_number": Stype.id,
    }
