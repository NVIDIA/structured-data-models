from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from sdm.stype import Stype

_INT64_MAX = 2**63 - 1


def to_nim_config(
    table: pa.Table,
    stypes: Mapping[str, Stype],
    *,
    primary_key: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    r"""Convert an Arrow table to NIM schema and Arrow data fragments."""
    if table.num_columns == 0:
        raise ValueError(
            "Cannot convert a table without columns to a NIM config"
        )
    table = _prepare_arrow_data(table)

    primary_key_columns = _primary_key_columns(primary_key)
    for column in primary_key_columns:
        if column not in table.column_names:
            raise ValueError(
                f"Expected primary key column {column!r} to exist"
            )

    if primary_key_columns:
        primary_key_values = {
            column: table[column].to_pylist() for column in primary_key_columns
        }
        seen_primary_keys: dict[tuple[Any, ...], int] = {}
        for row in range(table.num_rows):
            key = tuple(
                primary_key_values[column][row]
                for column in primary_key_columns
            )
            if any(_is_null_primary_key(value) for value in key):
                null_column = next(
                    column
                    for column, value in zip(primary_key_columns, key)
                    if _is_null_primary_key(value)
                )
                raise ValueError(
                    f"Expected primary key value at rows[{row}]"
                    f"[{null_column!r}] to be non-null"
                )
            if key in seen_primary_keys:
                first_row = seen_primary_keys[key]
                raise ValueError(
                    f"Expected primary key values to be unique; rows "
                    f"{first_row} and {row} both contain {key!r}"
                )
            seen_primary_keys[key] = row

    schema: dict[str, Any] = {
        "columns": {
            field.name: {
                "dtype": _nim_dtype(field.type),
                "stype": _nim_stype(stypes[field.name]),
                **(
                    {"nullable": False}
                    if field.name in primary_key_columns
                    else {}
                ),
            }
            for field in table.schema
        }
    }
    if primary_key is not None:
        schema["primary_key"] = (
            primary_key
            if isinstance(primary_key, str)
            else list(primary_key_columns)
        )

    return {"schema": schema, "data": table}


def _is_null_primary_key(value: Any) -> bool:
    return value is None


def _prepare_arrow_data(table: pa.Table) -> pa.Table:
    arrays: list[pa.Array | pa.ChunkedArray] = []
    for field in table.schema:
        column = table[field.name]
        dtype = field.type
        if pa.types.is_dictionary(dtype):
            dtype = dtype.value_type
            column = column.cast(dtype)

        target_dtype = _nim_arrow_dtype(dtype)
        if pa.types.is_uint64(dtype):
            maximum = pc.call_function("max", [column]).as_py()
            if maximum is not None and maximum > _INT64_MAX:
                raise ValueError(
                    f"Column {field.name!r} contains an unsigned integer "
                    "outside the supported int64 range"
                )
        if dtype != target_dtype:
            column = column.cast(target_dtype)

        if pa.types.is_floating(target_dtype):
            inf_mask = pc.call_function("is_inf", [column])
            if pc.call_function("any", [inf_mask]).as_py() is True:
                raise ValueError(
                    f"Expected finite numeric values in column {field.name!r}"
                )
            column = pc.call_function(
                "if_else",
                [
                    pc.call_function("is_nan", [column]),
                    pa.scalar(None, type=target_dtype),
                    column,
                ],
            )
        arrays.append(column)

    return pa.Table.from_arrays(arrays, names=table.column_names)


def _primary_key_columns(
    primary_key: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if primary_key is None:
        return ()
    if isinstance(primary_key, str):
        return (primary_key,)

    columns = tuple(primary_key)
    if not columns:
        raise ValueError("Expected 'primary_key' to be non-empty")
    if not all(isinstance(column, str) for column in columns):
        raise TypeError("Expected primary key columns to be strings")
    if len(columns) != len(set(columns)):
        raise ValueError("Expected primary key columns to be unique")
    return columns


def _nim_stype(stype: Stype) -> str:
    if stype == Stype.id:
        return "ID"
    if stype == Stype.datetime:
        return "timestamp"
    return stype.value


def _nim_dtype(dtype: pa.DataType) -> str:
    dtype = _nim_arrow_dtype(dtype)
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_int32(dtype):
        return "int32"
    if pa.types.is_int64(dtype):
        return "int64"
    if pa.types.is_float32(dtype):
        return "float32"
    if pa.types.is_float64(dtype):
        return "float64"
    if pa.types.is_string(dtype):
        return "string"
    if pa.types.is_timestamp(dtype):
        return "timestamp[us]"
    raise AssertionError(f"Unexpected normalized NIM dtype {dtype!r}")


def _nim_arrow_dtype(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(dtype):
        return _nim_arrow_dtype(dtype.value_type)
    if pa.types.is_boolean(dtype):
        return pa.bool_()
    if pa.types.is_signed_integer(dtype):
        return pa.int32() if dtype.bit_width <= 32 else pa.int64()
    if pa.types.is_unsigned_integer(dtype):
        return pa.int32() if dtype.bit_width < 32 else pa.int64()
    if pa.types.is_floating(dtype):
        return pa.float32() if dtype.bit_width <= 32 else pa.float64()
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_timestamp(dtype):
        return pa.timestamp("us")
    if pa.types.is_null(dtype):
        raise TypeError("Cannot infer a NIM dtype from an all-null column")
    raise TypeError(f"Unsupported NIM column type {dtype!r}")
