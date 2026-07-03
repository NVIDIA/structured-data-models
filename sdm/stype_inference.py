import importlib.util
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from sdm.stype import Stype

__all__ = [
    "infer_stypes",
]


def infer_stypes(table: Any) -> dict[str, Stype]:
    r"""Infer semantic column types for a pandas or Arrow table."""
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            return {
                str(column): _infer_pandas_stype(
                    table[column],
                    column=str(column),
                )
                for column in table.columns
            }

    if isinstance(table, pa.Table):
        return {
            field.name: _infer_arrow_stype(
                field.type,
                column=field.name,
            )
            for field in table.schema
        }

    if isinstance(table, Mapping):
        stypes: dict[str, Stype] = {}
        for column, value in table.items():
            if not isinstance(value, pa.ChunkedArray | pa.Array):
                raise TypeError(
                    "Expected an Arrow mapping column to be a 'pyarrow.Array' "
                    f"or 'pyarrow.ChunkedArray' (got '{type(value).__name__}')"
                )
            stypes[str(column)] = _infer_arrow_stype(
                value.type, column=str(column)
            )
        return stypes

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
    r"""Infer the :class:`~sdm.stype.Stype` of a pandas column.

    Boolean, string, object, and categorical dtypes map to
    :attr:`Stype.categorical`. Numeric dtypes map to :attr:`Stype.numerical`.

    Args:
        column_data: The ``pandas.Series`` to infer.
        column: The column name, used for error messages.
    """
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
    if column is None:
        raise TypeError(f"Unsupported pandas dtype '{dtype}'")
    raise TypeError(
        f"Unsupported pandas dtype for column '{column}': '{dtype}'"
    )


def _infer_arrow_stype(
    data_type: pa.DataType,
    *,
    column: str | None = None,
) -> Stype:
    r"""Infer the :class:`~sdm.stype.Stype` of an Arrow data type.

    Integer, floating-point, and decimal types map to
    :attr:`Stype.numerical`. String, boolean, and dictionary-encoded types
    map to :attr:`Stype.categorical`.

    Args:
        data_type: The Arrow data type to infer.
        column: The column name, used for error messages.
    """
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
    if column is None:
        raise TypeError(f"Unsupported Arrow type '{data_type}'")
    raise TypeError(
        f"Unsupported Arrow type for column '{column}': '{data_type}'"
    )
