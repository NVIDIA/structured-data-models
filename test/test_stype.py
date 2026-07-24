from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest
from sdm import Stype, infer_stypes
from sdm.testing import onlyCUDA


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


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

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


def test_text_detection_arrow() -> None:
    table = pa.table(
        {
            "review": pa.array(
                [
                    "the product broke after one week of use",
                    "excellent value and very fast shipping thanks",
                    "arrived damaged and support was unhelpful sadly",
                ],
                type=pa.string(),
            ),
            "color": pa.array(["red", "blue", "green"], type=pa.string()),
        }
    )

    assert infer_stypes(table, allowed_stypes={Stype.text}) == {
        "review": Stype.text,
        "color": Stype.categorical,
    }


def test_text_detection_pandas() -> None:
    df = pd.DataFrame(
        {
            "review": pd.Series(
                [
                    "the product broke after one week of use",
                    "excellent value and very fast shipping thanks",
                    "arrived damaged and support was unhelpful sadly",
                ],
                dtype="string",
            ),
            "color": pd.Series(["red", "blue", "green"], dtype="string"),
        }
    )

    assert infer_stypes(df, allowed_stypes={Stype.text}) == {
        "review": Stype.text,
        "color": Stype.categorical,
    }


def test_text_not_inferred_when_not_allowed() -> None:
    table = pa.table(
        {
            "review": pa.array(
                ["the product broke after one week of use"] * 3,
                type=pa.string(),
            ),
        }
    )

    assert infer_stypes(table) == {"review": Stype.categorical}


@onlyCUDA
def test_text_detection_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame(
        {
            "review": cudf.Series(
                [
                    "the product broke after one week of use",
                    "excellent value and very fast shipping thanks",
                    "arrived damaged and support was unhelpful sadly",
                ]
            ),
            "color": cudf.Series(["red", "blue", "green"]),
        }
    )

    assert infer_stypes(df, allowed_stypes={Stype.text}) == {
        "review": Stype.text,
        "color": Stype.categorical,
    }
