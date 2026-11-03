# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import re
import warnings
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

import sdm

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

    @classmethod
    def all(cls) -> tuple[Stype, ...]:
        r""":meta private:"""  # noqa: D415
        return (  # Explicit members avoid enum iteration in compiled regions.
            cls.numerical,
            cls.categorical,
            cls.datetime,
            cls.text,
            cls.id,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"


StypeLike: TypeAlias = Stype | str

# Semantic Type Inference #####################################################


@dataclass
class CategoricalConfig:
    r"""Thresholds for inferring numeric columns as :attr:`~Stype.categorical`.

    Args:
        min_rows: Minimum row count, including nulls.
        max_unique: Maximum distinct value count, including nulls.
    """

    min_rows: int = 151
    max_unique: int = 3


@dataclass
class TextConfig:
    r"""Thresholds for inferring string columns as :attr:`~Stype.text`.

    Args:
        min_unique: Minimum number of distinct strings, excluding nulls.
        min_unique_ratio: Minimum ratio of distinct strings, excluding nulls.
        min_avg_words: Minimum average number of words per distinct string.
    """

    min_unique: int = 200
    min_unique_ratio: float = 0.05
    min_avg_words: float = 3


@dataclass
class StypeConfig:
    r"""Semantic type inference thresholds.

    Args:
        categorical: Thresholds for inferring numeric columns as
            :attr:`~Stype.categorical`.
        text: Thresholds for inferring string columns as :attr:`~Stype.text`.
    """

    categorical: CategoricalConfig = field(default_factory=CategoricalConfig)
    text: TextConfig = field(default_factory=TextConfig)


config = StypeConfig()
Policy: TypeAlias = Literal["off", "infer", "drop"]


def infer_stypes(
    table: pa.Table | pd.DataFrame | cudf.DataFrame,
    overrides: Mapping[str, StypeLike] | None = None,
    *,
    text: Literal["off", "infer", "drop"] = "off",
    id: Literal["off", "infer", "drop"] = "off",
    unsupported: Literal["error", "warn", "drop"] = "error",
    config: StypeConfig | Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, StypeLike]:
    r"""Infer semantic types from raw data statistics.

    Semantic types are inferred as follows:

    * Integer, floating-point, and decimal columns are inferred as
      :attr:`~Stype.numerical` if they have more than three distinct values,
      and :attr:`~Stype.categorical` otherwise.
    * String, boolean and dictionary-encoded columns are inferred as
      :attr:`~Stype.categorical`.
    * Datetime columns are inferred as :attr:`~Stype.datetime`.

    Optionally, infer the following semantic types based on best-effort
    heuristics:

    * String columns are inferred as :attr:`~Stype.text` if they contain at
      least 200 distinct string values, a distinct non-null value ratio of at
      least 5%, and an average of at least three words per distinct value.
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
        config: Optional :class:`~sdm.stype.StypeConfig` overrides grouped by
            semantic type, *e.g.*, ``{"text": {"min_unique": 50}}``.

    Returns:
        Dictionary mapping column names to inferred semantic type.
    """
    overrides = overrides or {}

    if not isinstance(config, StypeConfig):
        config = replace(
            sdm.stype.config,
            **{
                name: replace(getattr(sdm.stype.config, name), **values)
                for name, values in (config or {}).items()
            },
        )

    fn = None
    columns: Iterable[tuple[Hashable, object]] | None = None
    if isinstance(table, pa.Table):
        fn = _infer_arrow_stype
        columns = zip(table.column_names, table.columns)

    if fn is None and importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        if isinstance(table, pd.DataFrame):
            fn = _infer_pandas_stype
            columns = table.items()

    if fn is None and importlib.util.find_spec("cudf") is not None:
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
        assert isinstance(name, str)
        if name in overrides:
            stypes[name] = Stype(overrides[name])
            continue

        try:
            stype = fn(name, column, text, id, config)
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
    array: object,
    text: Policy,
    id: Policy,
    config: StypeConfig,
) -> Stype | None:
    assert isinstance(array, pa.Array | pa.ChunkedArray)
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
        if _is_arrow_categorical(array, config.categorical):
            return Stype.categorical
        return Stype.numerical

    if pa.types.is_boolean(dtype) or pa.types.is_dictionary(dtype):
        return Stype.categorical

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        if text != "off" and _is_arrow_text(array, config.text):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported arrow type '{dtype}' for column {name!r}")


def _infer_pandas_stype(
    name: str,
    ser: object,
    text: Policy,
    id: Policy,
    config: StypeConfig,
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

    assert isinstance(ser, pd.Series)
    dtype = ser.dtype

    is_string = is_string_dtype(dtype)
    if is_object_dtype(dtype):
        is_string = infer_dtype(ser, skipna=True) == "string"

    if (
        id != "off"
        and (is_integer_dtype(dtype) or is_string)
        and _has_id_token(name)
    ):
        return None if id == "drop" else Stype.id

    if is_integer_dtype(dtype) or is_float_dtype(dtype):
        if _is_series_categorical(ser, config.categorical):
            return Stype.categorical
        return Stype.numerical

    if is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return Stype.categorical

    if is_string:
        if text != "off" and _is_arrow_text(
            array=pa.array(ser, from_pandas=True),
            config=config.text,
        ):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.PeriodDtype):
        return Stype.datetime

    raise TypeError(f"Unsupported pandas type '{dtype}' for column {name!r}")


def _infer_cudf_stype(
    name: str,
    ser: object,
    text: Policy,
    id: Policy,
    config: StypeConfig,
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

    assert isinstance(ser, cudf.Series)
    dtype = ser.dtype

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
        if _is_series_categorical(ser, config.categorical):
            return Stype.categorical
        return Stype.numerical

    if is_bool_dtype(dtype) or isinstance(dtype, cudf.CategoricalDtype):
        return Stype.categorical

    if is_string_dtype(dtype):
        if text != "off" and _is_cudf_text(ser, config.text):
            return None if text == "drop" else Stype.text
        return Stype.categorical

    if is_datetime64_any_dtype(dtype):
        return Stype.datetime

    raise TypeError(f"Unsupported cudf type '{dtype}' for column {name!r}")


# Tokenize strings on separators (non-letters/digits) and camelCase boundaries:
_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def _has_id_token(name: str) -> bool:
    return "id" in (word.lower() for word in _WORD_PATTERN.split(name))


def _is_arrow_text(
    array: pa.Array | pa.ChunkedArray,
    config: TextConfig,
) -> bool:
    if config.min_unique_ratio >= 1:
        return False

    num_values = len(array) - array.null_count
    if num_values == 0 or num_values < config.min_unique:
        return False

    unique = pc.call_function("unique", [array])
    unique = pc.call_function("drop_null", [unique])
    if len(unique) < config.min_unique:
        return False
    if len(unique) / num_values <= config.min_unique_ratio:
        return False

    tokens = pc.call_function("utf8_split_whitespace", [unique])
    words = pc.call_function("list_value_length", [tokens])
    avg_words = pc.call_function("mean", [words]).as_py()
    return avg_words >= config.min_avg_words


def _is_cudf_text(ser: cudf.Series, config: TextConfig) -> bool:
    if config.min_unique_ratio >= 1:
        return False

    num_values = ser.count()
    if num_values == 0 or num_values < config.min_unique:
        return False

    if (num_unique := ser.nunique(dropna=True)) < config.min_unique:
        return False
    if num_unique / num_values <= config.min_unique_ratio:
        return False

    unique = ser.dropna().unique()
    avg_words = unique.str.token_count().mean()
    return avg_words >= config.min_avg_words


def _is_arrow_categorical(
    array: pa.Array | pa.ChunkedArray,
    config: CategoricalConfig,
) -> bool:
    if config.max_unique < 2:
        return False
    if len(array) < config.min_rows:
        return False

    options = pc.CountOptions(mode="all")

    if len(array) > 1024 and config.max_unique < 1024:  # Exit early:
        tmp = array.slice(0, 1024)
        num_unique = pc.call_function("count_distinct", [tmp], options).as_py()
        if num_unique > config.max_unique:
            return False

    num_unique = pc.call_function("count_distinct", [array], options).as_py()
    return 1 < num_unique <= config.max_unique


def _is_series_categorical(
    ser: pd.Series | cudf.Series,
    config: CategoricalConfig,
) -> bool:
    if config.max_unique < 2:
        return False
    if len(ser) < config.min_rows:
        return False

    if len(ser) > 1024 and config.max_unique < 1024:  # Exit early:
        tmp = ser.iloc[:1024]
        if tmp.nunique(dropna=False) > config.max_unique:
            return False

    num_unique = ser.nunique(dropna=False)
    return 1 < num_unique <= config.max_unique
