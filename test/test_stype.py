from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pytest
import torch
from sdm import Stype, infer_stypes
from sdm.testing import onlyCUDA

if TYPE_CHECKING:
    import cudf


_BACKENDS = [
    "pandas",
    "arrow",
    pytest.param(
        "cudf",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA not available",
            ),
        ],
    ),
]

_TEST_DTYPE_SAMPLE_DATA = {
    "user_id": [1, 2],
    "age": [25, 31],
    "income": [1.0, 2.5],
    "name": ["a", "b"],
    "segment": ["x", "y"],
    "active": [True, False],
    "created_at": ["2026-01-01", "2026-01-02"],
}

_TEST_REVIEW_DATA = {
    "review": [
        f"review sentence number {i} with enough words" for i in range(10)
    ],
    "color": ["red", "blue", "green", "yellow", "black"] * 2,
}

# ``repeated`` holds too few distinct values, ``varied`` has enough distinct
# values and a unique ratio above 0.01:
_TEST_CARDINALITY_DATA = {
    "repeated": ["the product broke after one week of use"] * 100,
    "varied": [
        f"review sentence number {i} with enough words" for i in range(10)
    ]
    * 10,
}


@pytest.fixture(params=_BACKENDS)
def table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    if request.param == "pandas":
        return pd.DataFrame(_TEST_DTYPE_SAMPLE_DATA).astype(
            {
                "name": "string",
                "segment": "category",
                "created_at": "datetime64[ns]",
            }
        )

    if request.param == "arrow":
        return pa.table(_TEST_DTYPE_SAMPLE_DATA).cast(
            pa.schema(
                [
                    ("user_id", pa.int64()),
                    ("age", pa.int64()),
                    ("income", pa.float64()),
                    ("name", pa.string()),
                    ("segment", pa.dictionary(pa.int32(), pa.string())),
                    ("active", pa.bool_()),
                    ("created_at", pa.timestamp("s")),
                ]
            )
        )

    cudf = pytest.importorskip("cudf")

    return cudf.DataFrame(_TEST_DTYPE_SAMPLE_DATA).astype(
        {
            "segment": "category",
            "created_at": "datetime64[ns]",
        }
    )


def test_infer_stypes(table: pa.Table | pd.DataFrame | cudf.DataFrame) -> None:
    assert infer_stypes(table) == {
        "user_id": Stype.id,
        "age": Stype.numerical,
        "income": Stype.numerical,
        "name": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
        "created_at": Stype.datetime,
    }


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


def _string_table(
    backend: str,
    data: dict[str, list[str]],
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    if backend == "pandas":
        return pd.DataFrame(data).astype("string")

    if backend == "arrow":
        return pa.table(data)

    cudf = pytest.importorskip("cudf")

    return cudf.DataFrame(data)


@pytest.fixture(params=_BACKENDS)
def text_table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    return _string_table(request.param, _TEST_REVIEW_DATA)


@pytest.fixture(params=_BACKENDS)
def cardinality_table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    return _string_table(request.param, _TEST_CARDINALITY_DATA)


def test_text_detection(
    text_table: pa.Table | pd.DataFrame | cudf.DataFrame,
) -> None:
    assert infer_stypes(text_table, allow_text=True) == {
        "review": Stype.text,
        "color": Stype.categorical,
    }


@onlyCUDA
def test_text_inference_consistent_between_arrow_and_cudf() -> None:
    cudf = pytest.importorskip("cudf")
    values = [None, *_TEST_REVIEW_DATA["review"]]
    arrow_table = pa.table(
        {
            "review": values,
            "category": pa.array(values).dictionary_encode(),
        }
    )
    cudf_table = cudf.DataFrame(
        {
            "review": values,
            "category": cudf.Series(values, dtype="category"),
        }
    )

    arrow_stypes = infer_stypes(arrow_table, allow_text=True)
    cudf_stypes = infer_stypes(cudf_table, allow_text=True)

    assert arrow_stypes == {
        "review": Stype.text,
        "category": Stype.categorical,
    }
    assert cudf_stypes == arrow_stypes


def test_text_not_inferred_by_default(
    text_table: pa.Table | pd.DataFrame | cudf.DataFrame,
) -> None:
    assert infer_stypes(text_table) == {
        "review": Stype.categorical,
        "color": Stype.categorical,
    }


def test_text_not_inferred_below_unique_ratio(
    cardinality_table: pa.Table | pd.DataFrame | cudf.DataFrame,
) -> None:
    assert infer_stypes(cardinality_table, allow_text=True) == {
        "repeated": Stype.categorical,
        "varied": Stype.text,
    }


def test_text_not_inferred_below_min_unique_values() -> None:
    table = pa.table(
        {
            "status": [
                "customer accepted the promotional offer",
                "customer rejected the promotional offer",
            ]
            * 50,
        }
    )

    assert infer_stypes(table, allow_text=True) == {
        "status": Stype.categorical,
    }
