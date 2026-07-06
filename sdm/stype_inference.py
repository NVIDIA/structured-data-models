import importlib.util
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pyarrow as pa

from sdm.stype import Stype

__all__ = [
    "infer_stypes",
]

_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def infer_stypes(
    table: Any,
    *,
    id_columns: Iterable[str] | None = None,
) -> dict[str, Stype]:
    r"""Infer semantic column types for a pandas or Arrow table.

    Integer, floating-point, and decimal columns are inferred as
    :attr:`Stype.numerical`. String, boolean, and dictionary-encoded
    (categorical) columns are inferred as :attr:`Stype.categorical`. An
    integer or (non-dictionary) string column is inferred as
    :attr:`Stype.id` if its name contains ``id`` as a whole word (e.g.
    ``user_id``, ``userId``, ``id``, but not ``solid`` or ``covid``), since
    identifier columns cannot be told apart from ordinary numerical or
    categorical columns by dtype alone. Any other type raises a
    :class:`TypeError`. pandas inputs are routed through their Arrow schema
    so inference stays consistent across backends.

    The name-based :attr:`Stype.id` heuristic is best-effort. Use
    ``id_columns`` to state identifier columns explicitly instead of relying
    on naming convention.

    Args:
        table: A ``pandas.DataFrame``, ``pyarrow.Table``, or mapping of column
            names to ``pyarrow.Array``/``pyarrow.ChunkedArray`` values.
        id_columns: Column names to always infer as :attr:`Stype.id`,
            regardless of name or dtype. Takes precedence over the name-based
            heuristic.
    """
    id_column_set = (
        frozenset(id_columns) if id_columns is not None else frozenset()
    )

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            table = pa.Schema.from_pandas(table, preserve_index=False)

    if isinstance(table, pa.Table):
        table = table.schema

    if isinstance(table, pa.Schema):
        return {
            field.name: _infer_column_stype(
                field.name, field.type, id_column_set
            )
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
            stypes[str(column)] = _infer_column_stype(
                str(column), value.type, id_column_set
            )
        return stypes

    raise TypeError(
        "Expected 'table' to be a pandas DataFrame, pyarrow Table, or "
        f"mapping of Arrow arrays (got '{type(table).__name__}')"
    )


def _infer_column_stype(
    column: str,
    data_type: pa.DataType,
    id_columns: frozenset[str],
) -> Stype:
    if column in id_columns:
        return Stype.id
    if (
        pa.types.is_integer(data_type)
        or pa.types.is_string(data_type)
        or pa.types.is_large_string(data_type)
    ) and _looks_like_identifier(column):
        return Stype.id
    return _infer_arrow_stype(data_type, column=column)


def _looks_like_identifier(name: str) -> bool:
    words = (word.lower() for word in _WORD_PATTERN.split(name) if word)
    return "id" in words


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
