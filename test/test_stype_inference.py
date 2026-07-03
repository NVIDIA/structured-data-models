from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pytest
from sdm import Stype, infer_stypes
from sdm.stype_inference import _infer_arrow_stype


def test_infer_stypes_from_pandas_dataframe() -> None:
    df = pd.DataFrame(
        {
            "age": pd.Series([1, 2], dtype="int64"),
            "income": pd.Series([1.0, 2.5], dtype="float64"),
            "name": pd.Series(["a", "b"], dtype="string"),
            "city": pd.Series(["NY", None], dtype="object"),
            "segment": pd.Series(["x", "y"], dtype="category"),
            "active": pd.Series([True, False], dtype="bool"),
        }
    )

    assert infer_stypes(df) == {
        "age": Stype.numerical,
        "income": Stype.numerical,
        "name": Stype.categorical,
        "city": Stype.categorical,
        "segment": Stype.categorical,
        "active": Stype.categorical,
    }


def test_infer_stypes_from_arrow_table() -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "amount": pa.array(
                [Decimal("1.25"), None],
                type=pa.decimal128(8, 2),
            ),
            "ratio": pa.array([1.0, 2.5], type=pa.float32()),
            "name": pa.array(["a", "b"], type=pa.string()),
            "note": pa.array(["a", "b"], type=pa.large_string()),
            "active": pa.array([True, False], type=pa.bool_()),
            "code": pa.array(["x", "y"]).dictionary_encode(),
        }
    )

    assert infer_stypes(table) == {
        "id": Stype.numerical,
        "amount": Stype.numerical,
        "ratio": Stype.numerical,
        "name": Stype.categorical,
        "note": Stype.categorical,
        "active": Stype.categorical,
        "code": Stype.categorical,
    }


def test_infer_stypes_from_arrow_mapping() -> None:
    assert infer_stypes(
        {
            "id": pa.array([1, 2], type=pa.int32()),
            "name": pa.array(["a", "b"], type=pa.string()),
        }
    ) == {
        "id": Stype.numerical,
        "name": Stype.categorical,
    }


def test_pandas_unsupported_dtype_raises_clear_error() -> None:
    df = pd.DataFrame(
        {"created_at": pd.to_datetime(["2026-01-01", "2026-01-02"])}
    )

    with pytest.raises(
        TypeError,
        match=r"Unsupported Arrow type.*created_at.*timestamp",
    ):
        infer_stypes(df)


def test_arrow_unsupported_type_raises_clear_error() -> None:
    table = pa.table({"created_at": pa.array([0, 1], type=pa.timestamp("s"))})

    with pytest.raises(
        TypeError,
        match=r"Unsupported Arrow type.*created_at.*timestamp",
    ):
        infer_stypes(table)


def test_arrow_column_helper() -> None:
    assert _infer_arrow_stype(pa.int32()) == Stype.numerical
    assert (
        _infer_arrow_stype(pa.dictionary(pa.int32(), pa.string()))
        == Stype.categorical
    )

    with pytest.raises(TypeError, match="Unsupported Arrow type"):
        _infer_arrow_stype(pa.timestamp("s"))
