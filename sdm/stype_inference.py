from collections.abc import Mapping
from typing import Any

from sdm.stype import Stype

__all__ = [
    "infer_stypes",
]


def infer_stypes(table: Any) -> dict[str, Stype]:
    r"""Infer semantic column types for a pandas or Arrow table."""
    pandas_dataframe = _pandas_dataframe_type()
    if pandas_dataframe is not None and isinstance(table, pandas_dataframe):
        return {
            str(column): _infer_pandas_stype(
                table[column],
                column=str(column),
            )
            for column in table.columns
        }

    import pyarrow as pa

    if isinstance(table, pa.Table):
        return {
            field.name: _infer_arrow_stype(
                field.type,
                column=field.name,
            )
            for field in table.schema
        }

    if isinstance(table, Mapping):
        return {
            str(column): _infer_arrow_stype(
                _arrow_column_type(value),
                column=str(column),
            )
            for column, value in table.items()
        }

    raise TypeError(
        "Expected 'table' to be a pandas DataFrame, pyarrow Table, or "
        "mapping of Arrow arrays "
        f"(got '{type(table).__name__}')"
    )


def _infer_pandas_stype(
    column_data: Any,
    *,
    column: str | None = None,
) -> Stype:
    import pandas as pd
    from pandas.api.types import (
        is_bool_dtype,
        is_numeric_dtype,
        is_object_dtype,
        is_string_dtype,
    )

    dtype = column_data.dtype
    if (
        is_bool_dtype(dtype)
        or is_string_dtype(dtype)
        or is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    ):
        return Stype.categorical
    if is_numeric_dtype(dtype):
        return Stype.numerical
    raise TypeError(_unsupported_pandas_dtype_message(dtype, column=column))


def _infer_arrow_stype(
    data_type: Any,
    *,
    column: str | None = None,
) -> Stype:
    import pyarrow as pa

    if (
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    ):
        return Stype.numerical
    if (
        pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
        or pa.types.is_boolean(data_type)
        or pa.types.is_dictionary(data_type)
    ):
        return Stype.categorical
    raise TypeError(_unsupported_arrow_type_message(data_type, column=column))


def _pandas_dataframe_type() -> type[Any] | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.DataFrame


def _arrow_column_type(value: Any) -> Any:
    import pyarrow as pa

    if isinstance(value, pa.ChunkedArray | pa.Array):
        return value.type
    raise TypeError(
        "Expected an Arrow mapping column to be a 'pyarrow.Array' or "
        f"'pyarrow.ChunkedArray' (got '{type(value).__name__}')"
    )


def _unsupported_pandas_dtype_message(
    dtype: Any,
    *,
    column: str | None,
) -> str:
    if column is None:
        return f"Unsupported pandas dtype '{dtype}'"
    return f"Unsupported pandas dtype for column '{column}': '{dtype}'"


def _unsupported_arrow_type_message(
    data_type: Any,
    *,
    column: str | None,
) -> str:
    if column is None:
        return f"Unsupported Arrow type '{data_type}'"
    return f"Unsupported Arrow type for column '{column}': '{data_type}'"
