"""Semantic column types identifiers."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    import cudf
    import pandas as pd


class Stype(str, Enum):  # noqa: UP042
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
_TEXT_MIN_UNIQUE_VALUES = 10
_TEXT_MIN_UNIQUE_RATIO = 0.01
_TEXT_MIN_AVERAGE_WORD_COUNT = 3


def infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    overrides: Mapping[str, StypeLike] | None = None,
    allow_text: bool = False,
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred based on best-effort via the following
    heuristics:

    * Integer, floating-point, and decimal columns are inferred as
      ``numerical``.
    * String, boolean and dictionary-encoded columns are inferred as
      ``categorical``. When ``allow_text is True`` and they meet our text
      heuristics, they are inferred as ``text``.
    * Datetime columns are inferred as ``datetime``.
    * Integer or (non-dictionary) string columns are inferred as ``id`` if its
      name contains ``"id"`` as a whole word (*e.g.*, ``"user_id"``,
      ``"userId"``, ``"id"``, but not ``"solid"`` or ``"covid"``).

    Args:
        table: A :class:`pandas.DataFrame`, :class:`pyarrow.Table`, or
            :class:`cudf.DataFrame`.
        overrides: Optional semantic type overrides by column name.
        allow_text: Whether to enable experimental inference of string
            columns as :attr:`Stype.text`. Callers should trim ``table`` to a
            representative subset before enabling text inference, which
            requires at least 10 unique string values, a unique value ratio
            greater than 0.01, and an average of at least 3 words per unique
            value.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}

    schema: pa.Schema | None = None
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            # NOTE: Only text stype inference currently requires column data.
            if allow_text:
                table = pa.Table.from_pandas(table, preserve_index=False)
                schema = table.schema
            else:
                schema = pa.Schema.from_pandas(table, preserve_index=False)

    if importlib.util.find_spec("cudf") is not None:
        import cudf

        if isinstance(table, cudf.DataFrame):
            return {
                column: Stype(overrides[column])
                if column in overrides
                else _infer_cudf_stype(
                    column,
                    dtype,
                    table[column] if allow_text else None,
                )
                for column, dtype in table.dtypes.items()
            }

    if isinstance(table, pa.Table):
        schema = table.schema

    if schema is None:
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame', 'pyarrow.Table', "
            f"or 'cudf.DataFrame' (got {type(table).__name__!r})"
        )

    return {
        field.name: Stype(overrides[field.name])
        if field.name in overrides
        else _infer_arrow_stype(
            field.name,
            field.type,
            table[field.name] if allow_text else None,
        )
        for field in schema
    }


def _infer_arrow_stype(
    name: str,
    dtype: pa.DataType,
    column: pa.ChunkedArray | None = None,
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

    if pa.types.is_boolean(dtype) or pa.types.is_dictionary(dtype):
        return Stype.categorical

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        if column is not None and _is_text_stype_arrow(column):
            return Stype.text
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported Arrow type '{dtype}' for column {name!r}")


def _infer_cudf_stype(
    name: str,
    dtype: Any,
    column: cudf.Series | None = None,
) -> Stype:
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

    if is_bool_dtype(dtype) or isinstance(dtype, cudf.CategoricalDtype):
        return Stype.categorical

    if is_string_dtype(dtype):
        if column is not None and _is_text_stype_cudf(column):
            return Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cuDF type '{dtype}' for column {name!r}")


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))


def _is_text_stype_arrow(column: pa.ChunkedArray) -> bool:
    if (num_values := len(column) - column.null_count) == 0:
        return False

    values = pc.call_function("drop_null", [column])
    num_unique = pc.call_function("count_distinct", [values]).as_py()
    if num_unique < _TEXT_MIN_UNIQUE_VALUES:
        return False

    if num_unique / num_values <= _TEXT_MIN_UNIQUE_RATIO:
        return False

    # average word count per distinct value
    unique_values = pc.call_function("unique", [values])
    tokens = pc.call_function("utf8_split_whitespace", [unique_values])
    words = pc.call_function("list_value_length", [tokens])
    avg_words = pc.call_function("mean", [words]).as_py()
    return avg_words >= _TEXT_MIN_AVERAGE_WORD_COUNT


def _is_text_stype_cudf(column: cudf.Series) -> bool:
    if (num_values := column.count()) == 0:
        return False

    if (num_unique := column.nunique(dropna=True)) < _TEXT_MIN_UNIQUE_VALUES:
        return False

    if num_unique / num_values <= _TEXT_MIN_UNIQUE_RATIO:
        return False

    unique_values = column.dropna().unique()
    avg_words = unique_values.str.token_count().mean()
    return avg_words >= _TEXT_MIN_AVERAGE_WORD_COUNT
