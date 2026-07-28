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


def infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    overrides: Mapping[str, StypeLike] | None = None,
    text_sample_table: cudf.DataFrame | pd.DataFrame | pa.Table | None = None,
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred based on best-effort via the following
    heuristics:

    * Integer, floating-point, and decimal columns are inferred as
      ``numerical``.
    * String, boolean and dictionary-encoded columns are inferred as
      ``categorical``.
    * Datetime columns are inferred as ``datetime``.
    * Integer or (non-dictionary) string columns are inferred as ``id`` if its
      name contains ``"id"`` as a whole word (*e.g.*, ``"user_id"``,
      ``"userId"``, ``"id"``, but not ``"solid"`` or ``"covid"``).

    Args:
        table: A :class:`pandas.DataFrame`, :class:`pyarrow.Table`, or
            :class:`cudf.DataFrame`.
        overrides: Optional semantic type overrides by column name.
        text_sample_table: [Experimental feature only] Optional sample table
            used to infer :attr:`Stype.text` for string columns. When omitted,
            string columns are inferred as :attr:`Stype.categorical`. Text
            inference requires at least 10 unique string values, a unique
            value ratio greater than 0.01, and an average of at least 3 words
            per unique value.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}
    if text_sample_table is not None and len(text_sample_table) > 5_000:
        raise ValueError(
            "`text_sample_table` must contain at most 5,000 rows."
        )

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            table = pa.Schema.from_pandas(table, preserve_index=False)

    if importlib.util.find_spec("cudf") is not None:
        import cudf

        if isinstance(table, cudf.DataFrame):
            return {
                column: Stype(overrides[column])
                if column in overrides
                else _infer_cudf_stype(column, dtype, text_sample_table)
                for column, dtype in table.dtypes.items()
            }

    if isinstance(table, pa.Table):
        table = table.schema

    if not isinstance(table, pa.Schema):
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame', 'pyarrow.Table', "
            f"or 'cudf.DataFrame' (got '{type(table).__name__}')"
        )

    return {
        field.name: Stype(overrides[field.name])
        if field.name in overrides
        else _infer_arrow_stype(field.name, field.type, text_sample_table)
        for field in table
    }


def _infer_arrow_stype(
    name: str,
    dtype: pa.DataType,
    text_sample: pa.Table | pd.DataFrame | None = None,
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
        if text_sample is not None and _is_text_stype_arrow(name, text_sample):
            return Stype.text
        return Stype.categorical

    if pa.types.is_boolean(dtype) or pa.types.is_dictionary(dtype):
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported Arrow type '{dtype}' for column '{name}'")


def _infer_cudf_stype(
    name: str,
    dtype: Any,
    text_sample: cudf.DataFrame | None = None,
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
        if text_sample is not None and _is_text_stype_cudf(name, text_sample):
            return Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cuDF type '{dtype}' for column '{name}'")


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))


def _is_text_stype_arrow(
    name: str,
    text_sample: pa.Table | pd.DataFrame,
) -> bool:
    column = text_sample[name]
    values = (
        column.to_pylist()
        if isinstance(column, pa.ChunkedArray)
        else column.tolist()
    )

    num_values = 0
    unique_values: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            continue
        num_values += 1
        unique_values.add(value)

    if num_values == 0:
        return False

    num_unique = len(unique_values)
    if num_unique < 10:
        return False

    # cardinality
    unique_ratio = num_unique / num_values
    if unique_ratio <= 0.01:
        return False

    # average word count per distinct value
    avg_words = sum(len(value.split()) for value in unique_values) / len(
        unique_values
    )
    return avg_words >= 3


def _is_text_stype_cudf(name: str, text_sample: cudf.DataFrame) -> bool:
    column = text_sample[name]
    num_values = column.count()

    if num_values == 0:
        return False

    num_unique = column.nunique(dropna=True)
    if num_unique < 10:
        return False

    unique_ratio = num_unique / num_values
    if unique_ratio <= 0.01:
        return False

    unique_values = column.dropna().unique()
    avg_words = unique_values.str.token_count().mean()
    return avg_words >= 3
