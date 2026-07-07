"""Semantic column types identifiers."""

import importlib.util
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeAlias

import pyarrow as pa


class Stype(str, Enum):
    r"""The semantic type of a table column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into a feature space.
    Possible values are:

    Attributes:
        numerical: Numerical columns.
        categorical: Categorical columns.
        datetime: Date or date-time columns.
        id: Identifier values used to distinguish or link entities. Identifier
            columns are not used as model features by default.
    """

    numerical = "numerical"
    categorical = "categorical"
    datetime = "datetime"
    id = "id"


StypeLike: TypeAlias = Stype | str

# Semantic Type Inference #####################################################

# Tokenize strings on separators (non-letters/digits) and camelCase boundaries:
_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def infer_stypes(
    table: Any,
    overrides: Mapping[str, StypeLike] | None = None,
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred based on best-effort via the following
    heuristics:

    * Integer, floating-point, and decimal columns are inferred as
      ``numerical``.
    * String, boolean, and dictionary-encoded (categorical) columns are
      inferred as ``categorical``.
    * Datetime columns are inferred as ``datetime``.
    * Integer or (non-dictionary) string columns are inferred as ``id`` if its
      name contains ``"id"`` as a whole word (*e.g.*, ``"user_id"``,
      ``"userId"``, ``"id"``, but not ``"solid"`` or ``"covid"``).

    Args:
        table: A :class:`pandas.DataFrame` or :class:`pyarrow.Table`.
        overrides: Optional semantic type overrides by column name.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            table = pa.Schema.from_pandas(table, preserve_index=False)

    if isinstance(table, pa.Table):
        table = table.schema

    if not isinstance(table, pa.Schema):
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame' or 'pyarrow.Table' "
            f"(got '{type(table).__name__}')"
        )

    return {
        field.name: Stype(overrides[field.name])
        if field.name in overrides
        else _infer_stype(field.name, field.type)
        for field in table
    }


def _infer_stype(name: str, dtype: pa.DataType) -> Stype:
    if (
        pa.types.is_integer(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
    ) and "id" in (word.lower() for word in _WORD_PATTERN.split(name)):
        return Stype.id

    if (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
    ):
        return Stype.numerical

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
