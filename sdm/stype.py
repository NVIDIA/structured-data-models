"""Semantic column types identifiers."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import pyarrow as pa

if TYPE_CHECKING:
    import cudf
    import pandas as pd


class Stype(str, Enum):
    r"""The semantic type of a table column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into a feature space.
    Possible values are:

    Attributes:
        numerical: Numerical columns.
        categorical: Categorical columns.
        datetime: Date or date-time columns.
        text: Text columns.
        id: Identifier values used to distinguish or link entities. Identifier
            columns are not used as model features by default.
    """

    numerical = "numerical"
    categorical = "categorical"
    datetime = "datetime"
    text = "text"
    id = "id"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


StypeLike: TypeAlias = Stype | str

# Semantic Type Inference #####################################################

# Tokenize strings on separators (non-letters/digits) and camelCase boundaries:
_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")

_TEXT_SAMPLE_SIZE = 1_000
_TEXT_MIN_UNIQUE = 20
_TEXT_UNIQUE_RATIO = 0.5


def infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    overrides: Mapping[str, StypeLike] | None = None,
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred based on best-effort via the following
    heuristics:

    * Integer, floating-point, and decimal columns are inferred as
      ``numerical``.
    * Boolean and dictionary-encoded columns are inferred as
      ``categorical``.
    * String columns are inferred as ``text`` if their first 1,000 rows
      contain at least 20 unique non-null values and more than 50% of
      the non-null values are unique, and as ``categorical`` otherwise.
    * Datetime columns are inferred as ``datetime``.
    * Integer or (non-dictionary) string columns are inferred as ``id`` if its
      name contains ``"id"`` as a whole word (*e.g.*, ``"user_id"``,
      ``"userId"``, ``"id"``, but not ``"solid"`` or ``"covid"``).

    Args:
        table: A :class:`pandas.DataFrame`, :class:`pyarrow.Table`, or
            :class:`cudf.DataFrame`.
        overrides: Optional semantic type overrides by column name.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            # Infer dtypes on the full frame, but only convert the value
            # sample needed by the text cardinality rule:
            table = pa.Table.from_pandas(
                df=table.iloc[:_TEXT_SAMPLE_SIZE],
                schema=pa.Schema.from_pandas(table, preserve_index=False),
                preserve_index=False,
            )

    if importlib.util.find_spec("cudf") is not None:
        import cudf

        if isinstance(table, cudf.DataFrame):
            return {
                column: Stype(overrides[column])
                if column in overrides
                else _infer_cudf_stype(column, dtype, table[column])
                for column, dtype in table.dtypes.items()
            }

    if not isinstance(table, pa.Table):
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame', 'pyarrow.Table', "
            f"or 'cudf.DataFrame' (got '{type(table).__name__}')"
        )

    return {
        field.name: Stype(overrides[field.name])
        if field.name in overrides
        else _infer_arrow_stype(field.name, field.type, table.column(i))
        for i, field in enumerate(table.schema)
    }


def _infer_arrow_stype(
    name: str,
    dtype: pa.DataType,
    column: pa.ChunkedArray,
) -> Stype:
    if (
        pa.types.is_integer(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
    ) and _has_id_token(name):
        return Stype.id

    if (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
    ):
        return Stype.numerical

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        sample = column.slice(0, _TEXT_SAMPLE_SIZE).drop_null()
        if _has_text_cardinality(len(sample.unique()), len(sample)):
            return Stype.text
        return Stype.categorical

    if pa.types.is_boolean(dtype) or pa.types.is_dictionary(dtype):
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported Arrow type '{dtype}' for column '{name}'")


def _infer_cudf_stype(name: str, dtype: Any, column: Any) -> Stype:
    import cudf
    from cudf.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_decimal_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_string_dtype,
    )

    if (is_integer_dtype(dtype) or is_string_dtype(dtype)) and _has_id_token(
        name
    ):
        return Stype.id

    if (
        is_integer_dtype(dtype)
        or is_float_dtype(dtype)
        or is_decimal_dtype(dtype)
    ):
        return Stype.numerical

    if is_string_dtype(dtype) and not isinstance(dtype, cudf.CategoricalDtype):
        sample = column.iloc[:_TEXT_SAMPLE_SIZE].dropna()
        if _has_text_cardinality(sample.nunique(), len(sample)):
            return Stype.text
        return Stype.categorical

    if is_bool_dtype(dtype) or isinstance(dtype, cudf.CategoricalDtype):
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cuDF type '{dtype}' for column '{name}'")


def _has_text_cardinality(num_unique: int, num_valid: int) -> bool:
    if num_unique < _TEXT_MIN_UNIQUE:
        return False
    return num_unique > _TEXT_UNIQUE_RATIO * num_valid


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))
