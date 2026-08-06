from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pytest
import torch

from sdm import Stype, infer_stypes

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


@pytest.fixture(params=_BACKENDS)
def table(
    request: pytest.FixtureRequest,
) -> pa.Table | pd.DataFrame | cudf.DataFrame:
    data = {
        "user_id": [1, 2],
        "age": [25, 31],
        "income": [1.0, 2.5],
        "name": ["a", "b"],
        "segment": ["x", "y"],
        "active": [True, False],
        "created_at": ["2026-01-01", "2026-01-02"],
        "review": [f"review sentence number {i}" for i in range(2)],
    }

    if request.param == "pandas":
        return pd.DataFrame(data).astype(
            {
                "segment": "category",
                "created_at": "datetime64[ns]",
            }
        )

    if request.param == "arrow":
        return pa.table(data).cast(
            pa.schema(
                [
                    ("user_id", pa.int64()),
                    ("age", pa.int64()),
                    ("income", pa.float64()),
                    ("name", pa.string()),
                    ("segment", pa.dictionary(pa.int32(), pa.string())),
                    ("active", pa.bool_()),
                    ("created_at", pa.timestamp("s")),
                    ("review", pa.string()),
                ]
            )
        )

    cudf = pytest.importorskip("cudf")

    return cudf.DataFrame(data).astype(
        {
            "segment": "category",
            "created_at": "datetime64[ns]",
        }
    )


def test_infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sdm.stype._TEXT_MIN_UNIQUE_VALUES", 2)

    assert infer_stypes(table, with_id=True, with_text=True) == {
        "user_id": Stype.id,
        "age": Stype.numerical,
        "income": Stype.numerical,
        "name": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
        "created_at": Stype.datetime,
        "review": Stype.text,
    }


def test_infer_stypes_pandas_object_strings() -> None:
    table = pd.DataFrame(
        {
            "city": pd.Series(["NYC", "LA"], dtype=object),
            "user_id": pd.Series(["a", "b"], dtype=object),
            "segment_id": pd.Series(["x", "y"], dtype="category"),
        }
    )

    assert infer_stypes(table, with_id=True) == {
        "city": Stype.categorical,
        "user_id": Stype.id,
        "segment_id": Stype.categorical,
    }


def test_id_detection() -> None:
    table = pa.table(
        {
            "user_id": pa.array([1, 2], type=pa.int64()),
            "userId": pa.array(["a", "b"], type=pa.string()),
            "order_id_hash": pa.array([1, 2], type=pa.int32()),
            "is_valid": pa.array(["a", "b"], type=pa.string()),
            "solid": pa.array([1, 2], type=pa.int64()),
            "covid_cases": pa.array([1, 2], type=pa.int64()),
        }
    )

    assert infer_stypes(table, with_id=True) == {
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
