from __future__ import annotations

import importlib.util
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    import cudf
    import pandas as pd


class Stype(StrEnum):
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


Policy: TypeAlias = Literal["off", "infer", "drop"]


def infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    overrides: Mapping[str, StypeLike] | None = None,
    *,
    text: Policy = "off",
    id: Policy = "off",
    unsupported: Literal["error", "warn", "drop"] = "error",
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred as follows:

    * Integer, floating-point, and decimal columns are inferred as
      :attr:`~Stype.numerical`.
    * String, boolean and dictionary-encoded columns are inferred as
      :attr:`~Stype.categorical`.
    * Datetime columns are inferred as :attr:`~Stype.datetime`.

    Optionally, infer the following semantic types based on best-effort
    heuristics:

    * String columns are inferred as :attr:`~Stype.text` if they contain at
      least 200 distinct string values, a distinct non-null value ratio of at
      least 5%, and an average of at least ``3`` words per distinct value.
    * Integer or (non-dictionary) string columns are inferred as
      :attr:`~Stype.id` if its name contains ``"id"`` as a whole word
      (*e.g.*, ``"user_id"``, ``"userId"``, ``"id"``, but not ``"solid"`` or
      ``"covid"``).

    Args:
        table: A :class:`pandas.DataFrame`, :class:`pyarrow.Table`, or
            :class:`cudf.DataFrame`.
        overrides: Optional semantic type overrides by column name.
        text: The :attr:`~Stype.text` detection policy.
            ``"off"`` disables :attr:`~Stype.text` column detection.
            ``"infer"`` includes inferred :attr:`~Stype.text` columns.
            ``"drop"`` omits inferred :attr:`~Stype.text` columns.
        id: The :attr:`~Stype.id` detection policy.
            ``"off"`` disables :attr:`~Stype.id` column detection.
            ``"infer"`` includes inferred :attr:`~Stype.id` columns.
            ``"drop"`` omits inferred :attr:`~Stype.id` columns.
        unsupported: How to handle unsupported dtypes.
            ``"error"`` raises a :class:`TypeError`.
            ``"warn"`` emits a warning and omits the column.
            ``"drop"`` omits the column silently.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}

    fn: Callable[[str, Any, Policy, Policy], Stype | None] | None = None
    columns: Iterable[tuple[str, Any]] | None = None
    if isinstance(table, pa.Table):
        fn = _infer_arrow_stype
        columns = zip(table.column_names, table.columns)
    elif importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            fn = _infer_pandas_stype
            columns = table.items()
    elif importlib.util.find_spec("cudf") is not None:
        import cudf

        if isinstance(table, cudf.DataFrame):
            fn = _infer_cudf_stype
            columns = table.items()

    if fn is None or columns is None:
        raise TypeError(
            f"Expected input to be a 'pandas.DataFrame', 'pyarrow.Table', "
            f"or 'cudf.DataFrame' (got {type(table).__name__!r})"
        )

    stypes = {}
    unsupported_columns: list[str] = []
    for name, column in columns:
        if name in overrides:
            stypes[name] = Stype(overrides[name])
            continue

        try:
            stype = fn(name, column, text, id)
        except TypeError:
            if unsupported == "error":
                raise
            if unsupported == "warn":
                unsupported_columns.append(name)
            continue

        if stype is not None:
            stypes[name] = stype

    if len(unsupported_columns) == 1:
        warnings.warn(
            f"Dropping unsupported column {unsupported_columns[0]!r}",
            stacklevel=2,
        )
    elif len(unsupported_columns) > 1:
        warnings.warn(
            f"Dropping unsupported columns {unsupported_columns}",
            stacklevel=2,
        )

    return stypes


# Helpers #####################################################################


def _infer_arrow_stype(
    name: str,
    array: pa.Array | pa.ChunkedArray,
    text: Policy,
    id: Policy,
) -> Stype | None:

    dtype = array.type

    if (
        id != "off"
        and (
            pa.types.is_integer(dtype)
            or pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
        )
        and _has_id_token(name)
    ):
        return None if id == "drop" else Stype.id

    if (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
    ):
        return Stype.numerical

    if pa.types.is_boolean(dtype) or pa.types.is_dictionary(dtype):
        return Stype.categorical

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        if text != "off" and _is_arrow_text(array):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported arrow type '{dtype}' for column {name!r}")


def _infer_pandas_stype(
    name: str,
    column: pd.Series,
    text: Policy,
    id: Policy,
) -> Stype | None:
    import pandas as pd
    from pandas.api.types import (
        infer_dtype,
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_object_dtype,
        is_string_dtype,
    )

    dtype = column.dtype

    is_string = is_string_dtype(dtype)
    if is_object_dtype(dtype):
        is_string = infer_dtype(column, skipna=True) == "string"

    if (
        id != "off"
        and (is_integer_dtype(dtype) or is_string)
        and _has_id_token(name)
    ):
        return None if id == "drop" else Stype.id

    if is_integer_dtype(dtype) or is_float_dtype(dtype):
        return Stype.numerical

    if is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return Stype.categorical

    if is_string:
        if text != "off" and _is_arrow_text(
            pa.array(column, from_pandas=True)
        ):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported pandas type '{dtype}' for column {name!r}")


def _infer_cudf_stype(
    name: str,
    column: cudf.Series,
    text: Policy,
    id: Policy,
) -> Stype | None:
    import cudf
    from cudf.api.types import (
        is_bool_dtype,
        is_datetime64_any_dtype,
        is_decimal_dtype,
        is_float_dtype,
        is_integer_dtype,
        is_string_dtype,
    )

    dtype = column.dtype

    if (
        id != "off"
        and (is_integer_dtype(dtype) or is_string_dtype(dtype))
        and _has_id_token(name)
    ):
        return None if id == "drop" else Stype.id

    if (
        is_integer_dtype(dtype)
        or is_float_dtype(dtype)
        or is_decimal_dtype(dtype)
    ):
        return Stype.numerical

    if is_bool_dtype(dtype) or isinstance(dtype, cudf.CategoricalDtype):
        return Stype.categorical

    if is_string_dtype(dtype):
        if text != "off" and _is_cudf_text(column):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cudf type '{dtype}' for column {name!r}")


# Tokenize strings on separators (non-letters/digits) and camelCase boundaries:
_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))


_TEXT_MIN_UNIQUE_VALUES = 200
_TEXT_MIN_UNIQUE_RATIO = 0.05
_TEXT_MIN_AVERAGE_WORD_COUNT = 3


def _is_arrow_text(array: pa.Array | pa.ChunkedArray) -> bool:
    if (num_values := len(array) - array.null_count) == 0:
        return False

    array = pc.call_function("drop_null", [array])
    num_unique = pc.call_function("count_distinct", [array]).as_py()
    if num_unique < _TEXT_MIN_UNIQUE_VALUES:
        return False
    if num_unique / num_values <= _TEXT_MIN_UNIQUE_RATIO:
        return False

    unique = pc.call_function("unique", [array])
    tokens = pc.call_function("utf8_split_whitespace", [unique])
    words = pc.call_function("list_value_length", [tokens])
    avg_words = pc.call_function("mean", [words]).as_py()
    return avg_words >= _TEXT_MIN_AVERAGE_WORD_COUNT


def _is_cudf_text(column: cudf.Series) -> bool:
    if (num_values := column.count()) == 0:
        return False

    if (num_unique := column.nunique(dropna=True)) < _TEXT_MIN_UNIQUE_VALUES:
        return False
    if num_unique / num_values <= _TEXT_MIN_UNIQUE_RATIO:
        return False

    unique = column.dropna().unique()
    avg_words = unique.str.token_count().mean()
    return avg_words >= _TEXT_MIN_AVERAGE_WORD_COUNT
