"""Semantic column types identifiers."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
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
    allowed_stypes: set[Stype] | None = None,
    seed: int | None = None,
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
        allowed_stypes: Optional set of semantic types eligible for
            inference. Inferring :attr:`Stype.text` requires sampling
            column values, so text is only considered when
            :attr:`Stype.text` is included; otherwise string columns fall
            back to :attr:`Stype.categorical`.
        seed: Optional seed for the row sampling used by text inference.
            Tables of at most 10,000 rows are inspected in full, and are
            unaffected by this argument.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}
    sample = None

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            if allowed_stypes and Stype.text in allowed_stypes:
                sample = table.sample(
                    n=min(10000, len(table)),
                    random_state=seed,
                )
            table = pa.Schema.from_pandas(table, preserve_index=False)

    if importlib.util.find_spec("cudf") is not None:
        import cudf

        if isinstance(table, cudf.DataFrame):
            if allowed_stypes and Stype.text in allowed_stypes:
                sample = table.sample(
                    n=min(10000, len(table)),
                    random_state=seed,
                )
            return {
                column: Stype(overrides[column])
                if column in overrides
                else _infer_cudf_stype(column, dtype, sample)
                for column, dtype in table.dtypes.items()
            }

    if isinstance(table, pa.Table):
        if allowed_stypes and Stype.text in allowed_stypes:
            n = min(10_000, table.num_rows)
            rng = np.random.default_rng(seed)
            idx = rng.choice(table.num_rows, size=n, replace=False)
            sample = table.take(idx)
        table = table.schema

    if not isinstance(table, pa.Schema):
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame', 'pyarrow.Table', "
            f"or 'cudf.DataFrame' (got '{type(table).__name__}')"
        )

    return {
        field.name: Stype(overrides[field.name])
        if field.name in overrides
        else _infer_arrow_stype(field.name, field.type, sample)
        for field in table
    }


def _infer_arrow_stype(
    name: str,
    dtype: pa.DataType,
    sample: pa.Table | pd.DataFrame | None = None,
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

    if (
        pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
    ) and sample is not None:
        if _is_text_stype_arrow(name, sample):
            return Stype.text
        return Stype.categorical

    if (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_boolean(dtype)
        or pa.types.is_dictionary(dtype)
    ):
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported Arrow type '{dtype}' for column '{name}'")


def _infer_cudf_stype(
    name: str,
    dtype: Any,
    sample: cudf.DataFrame | None = None,
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

    if is_string_dtype(dtype) and sample is not None:
        if _is_text_stype_cudf(name, sample):
            return Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cuDF type '{dtype}' for column '{name}'")


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))


def _is_text_stype_arrow(
    name: str,
    table: pa.Table | pd.DataFrame,
) -> bool:
    column = table[name]
    values = (
        column.to_pylist()
        if isinstance(column, pa.ChunkedArray)
        else column.tolist()
    )
    values = [v for v in values if isinstance(v, str)]  # drop nulls / non-str
    if not values:
        return False

    # cardinality
    unique_ratio = len(set(values)) / len(values)
    # average word count per cell
    word_counts = [len(value.split()) for value in values]
    avg_words = sum(word_counts) / len(word_counts)

    return unique_ratio > 0.01 and avg_words >= 3


def _is_text_stype_cudf(name: str, table: cudf.DataFrame) -> bool:
    column = table[name]
    values = column.dropna()

    if values.empty:
        return False

    unique_ratio = values.nunique() / len(values)
    word_counts = values.str.token_count()
    avg_words = word_counts.mean()

    return unique_ratio > 0.01 and avg_words >= 3
