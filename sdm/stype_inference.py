import importlib.util
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from sdm.stype import Stype

__all__ = [
    "infer_stypes",
]


def infer_stypes(table: Any) -> dict[str, Stype]:
    r"""Infer semantic column types for a pandas or Arrow table.

    Integer, floating-point, and decimal columns are inferred as
    :attr:`Stype.numerical`. String, boolean, and dictionary-encoded
    (categorical) columns are inferred as :attr:`Stype.categorical`. Any other
    type raises a :class:`TypeError`. pandas inputs are routed through their
    Arrow schema so inference stays consistent across backends.

    Args:
        table: A ``pandas.DataFrame``, ``pyarrow.Table``, or mapping of column
            names to ``pyarrow.Array``/``pyarrow.ChunkedArray`` values.
    """
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            table = pa.Schema.from_pandas(table, preserve_index=False)

    if isinstance(table, pa.Table):
        table = table.schema

    if isinstance(table, pa.Schema):
        return {
            field.name: _infer_arrow_stype(field.type, column=field.name)
            for field in table
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
        f"mapping of Arrow arrays (got '{type(table).__name__}')"
    )


def _infer_arrow_stype(
    data_type: pa.DataType,
    *,
    column: str | None = None,
) -> Stype:
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
