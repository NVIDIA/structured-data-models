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

_TEST_REVIEW_SAMPLE_DATA = {
    "review": [
        "the product broke after one week of use",
        "excellent value and very fast shipping thanks",
        "arrived damaged and support was unhelpful sadly",
    ],
    "color": ["red", "blue", "green"],
}

# ``repeated`` holds a unique ratio of exactly 0.01, ``varied`` of 0.02:
_TEST_CARDINALITY_SAMPLE_DATA = {
    "repeated": ["the product broke after one week of use"] * 100,
    "varied": [
        "the product broke after one week of use",
        "excellent value and very fast shipping thanks",
    ]
    * 50,
}

# Exceeds the 10,000 row sampling threshold, and holds a unique ratio close
# enough to the text cutoff that unsampled draws disagree on the stype:
_TEST_BORDERLINE_SAMPLE_DATA = {
    "review": [
        f"sentence number {i} of this borderline column" for i in range(200)
    ]
    + ["the product broke after one week of use"] * 19800,
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
    df = pd.DataFrame({"city": pd.Series(["NY", None], dtype="object")})

    assert infer_stypes(df) == {"city": Stype.categorical}


def test_from_arrow() -> None:
    table = pa.table(
        {
            "amount": pa.array([Decimal("1.25"), None]),
            "ratio": pa.array([1.0, 2.5], type=pa.float32()),
            "note": pa.array(["a", "b"], type=pa.large_string()),
        }
    )

    assert infer_stypes(table) == {
        "amount": Stype.numerical,
        "ratio": Stype.numerical,
        "note": Stype.categorical,
    }


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    df = cudf.DataFrame(
        {
            "age": cudf.Series([25, 31], dtype="int32"),
            "amount": cudf.Series(
                [Decimal("1.25"), Decimal("2.50")],
                dtype=cudf.Decimal64Dtype(8, 2),
            ),
        }
    )

    assert infer_stypes(df) == {
        "age": Stype.numerical,
        "amount": Stype.numerical,
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
    return _string_table(request.param, _TEST_REVIEW_SAMPLE_DATA)


@pytest.fixture(params=_BACKENDS)
def cardinality_table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    return _string_table(request.param, _TEST_CARDINALITY_SAMPLE_DATA)


@pytest.fixture(params=_BACKENDS)
def borderline_table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    return _string_table(request.param, _TEST_BORDERLINE_SAMPLE_DATA)


@pytest.mark.parametrize(
    "allowed_stypes",
    [
        {Stype.text},
        {Stype.categorical, Stype.text},
    ],
)
def test_text_detection(
    text_table: pa.Table | pd.DataFrame | cudf.DataFrame,
    allowed_stypes: set[Stype],
) -> None:
    assert infer_stypes(text_table, allowed_stypes=allowed_stypes) == {
        "review": Stype.text,
        "color": Stype.categorical,
    }


@pytest.mark.parametrize(
    "allowed_stypes",
    [
        None,
        set(),
        {Stype.categorical},
    ],
)
def test_text_not_inferred_when_not_allowed(
    text_table: pa.Table | pd.DataFrame | cudf.DataFrame,
    allowed_stypes: set[Stype] | None,
) -> None:
    assert infer_stypes(text_table, allowed_stypes=allowed_stypes) == {
        "review": Stype.categorical,
        "color": Stype.categorical,
    }


def test_text_not_inferred_below_unique_ratio(
    cardinality_table: pa.Table | pd.DataFrame | cudf.DataFrame,
) -> None:
    assert infer_stypes(cardinality_table, allowed_stypes={Stype.text}) == {
        "repeated": Stype.categorical,
        "varied": Stype.text,
    }


def test_seeded_text_inference_is_reproducible(
    borderline_table: pa.Table | pd.DataFrame | cudf.DataFrame,
) -> None:
    stypes = [
        infer_stypes(
            borderline_table,
            allowed_stypes={Stype.text},
            seed=0,
        )
        for _ in range(5)
    ]

    assert all(stype == stypes[0] for stype in stypes)
