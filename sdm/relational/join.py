# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import math
from collections.abc import Sequence
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import ColumnarTensor, NullableTensor, StringTensor, TableTensor
from sdm._warnings import warn_once
from sdm.tensor.io import arrow_as_tensor, to_cudf
from sdm.tensor.var_len import _clone

PREFIX = "sdm_internal"
LEFT_ROW_ID = f"__{PREFIX}_left_row_id__"
RIGHT_ROW_ID = f"__{PREFIX}_right_row_id__"


def join_index(
    left_table: TableTensor,
    right_table: TableTensor,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    how: Literal["inner"] = "inner",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return row-index pairs for matching rows in two tables.

    Args:
        left_table: The left table.
        right_table: The right table.
        left_keys: Column names from ``left_table`` used as join keys.
        right_keys: Column names from ``right_table`` used as join keys.
        how: The join type.
        dtype: The dtype.
        device: The device.

    Returns:
        ``(left_index, right_index)`` pair with one entry per matched row.
    """
    # Keep Arrow/cuDF opaque; only the surrounding tensor work is compiled.
    if torch.compiler.is_compiling() and all(
        table._column_to_loc[key][0] == "id"
        for table, keys in ((left_table, left_keys), (right_table, right_keys))
        for key in keys
    ):
        columns = [
            getattr(table.id, f"_column_{table._column_to_loc[key][1]}")
            for table, keys in (
                (left_table, left_keys),
                (right_table, right_keys),
            )
            for key in keys
        ]
        if all(
            type(column) is Tensor
            or isinstance(column, (NullableTensor, StringTensor))
            for column in columns
        ):
            assert how == "inner"
            leaves, kinds, metadata = _pack_columns(columns)
            return _join_indices(
                leaves,
                kinds,
                metadata,
                len(left_keys),
                dtype or torch.long,
                torch.device(device)
                if device is not None
                else left_table.device,
            )
    return _join_index_eager(
        left_table, right_table, left_keys, right_keys, how, dtype, device
    )


@torch.compiler.disable
def _join_index_eager(
    left_table: TableTensor,
    right_table: TableTensor,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    how: Literal["inner"] = "inner",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    assert how == "inner"

    if left_table.device != right_table.device:
        raise RuntimeError(
            "Expected 'left_table' and 'right_table' to be on the same device "
            f"(got '{left_table.device}' and '{right_table.device}')"
        )

    dtype = dtype or torch.long
    device = device or left_table.device

    left_table = left_table[list(left_keys)]
    right_table = right_table[list(right_keys)]
    left_rows = math.prod(left_table.size()[:-1])
    right_rows = math.prod(right_table.size()[:-1])

    if max(left_rows, right_rows) - 1 > torch.iinfo(dtype).max:
        raise ValueError(
            f"Creating row indices up to {max(left_rows, right_rows) - 1:,} "
            f"in '{dtype}' would overflow"
        )

    backend: Literal["arrow", "cudf"] = "arrow"
    if left_table.is_cuda:
        if importlib.util.find_spec("cudf") is not None:
            backend = "cudf"
        else:
            warn_once(
                key="missing-cudf-join",
                message=(
                    "Falling back to a CPU-based join because cuDF is not "
                    "installed. Install cuDF to enable faster CUDA-based "
                    "joins without device synchronization."
                ),
            )

    if backend == "arrow":
        left = left_table.to_arrow().append_column(
            LEFT_ROW_ID,
            pa.array(torch.arange(left_rows, dtype=dtype).numpy()),
        )
        right = right_table.to_arrow().append_column(
            RIGHT_ROW_ID,
            pa.array(torch.arange(right_rows, dtype=dtype).numpy()),
        )

        left = dropna(left, subset=left_keys)
        right = dropna(right, subset=right_keys)

        right_schema = right.schema
        for left_key, right_key in zip(left_keys, right_keys):
            left_type = left[left_key].type
            if left_type != right[right_key].type:
                index = right.schema.get_field_index(right_key)
                field = right.schema.field(index).with_type(left_type)
                right_schema = right_schema.set(index, field)
        if right_schema != right.schema:
            right = right.cast(right_schema, safe=True)

        joined = left.join(
            right,
            keys=left_keys,
            right_keys=right_keys,
            join_type=how,
        )

        return (
            arrow_as_tensor(joined[LEFT_ROW_ID], device=device),
            arrow_as_tensor(joined[RIGHT_ROW_ID], device=device),
        )

    assert backend == "cudf"
    with torch.cuda.device(left_table.device):
        left = left_table.to_cudf()
        left[LEFT_ROW_ID] = to_cudf(
            torch.arange(left_rows, dtype=dtype, device=left_table.device)
        )
        right = right_table.to_cudf()
        right[RIGHT_ROW_ID] = to_cudf(
            torch.arange(right_rows, dtype=dtype, device=right_table.device)
        )

        joined = left.dropna(subset=list(left_keys)).merge(
            right.dropna(subset=list(right_keys)),
            left_on=left_keys,
            right_on=right_keys,
            how=how,
        )

        return (
            torch.as_tensor(joined[LEFT_ROW_ID]).to(device),
            torch.as_tensor(joined[RIGHT_ROW_ID]).to(device),
        )


def dropna(
    table: pa.Table,
    subset: Sequence[str] | None = None,
) -> pa.Table:
    r"""Drop rows with null or NaN values in the given columns."""
    columns = table.column_names if subset is None else subset

    mask = None
    for name in columns:
        column = table[name]

        _mask = None
        if column.null_count > 0:
            _mask = column.is_valid()

        if pa.types.is_floating(column.type):
            __mask = pc.call_function(
                "invert", [pc.call_function("is_nan", [column])]
            )
            if _mask is not None:
                _mask = pc.call_function("and_kleene", [_mask, __mask])
            else:
                _mask = __mask

        if _mask is None:
            continue

        if mask is not None:
            mask = pc.call_function("and_kleene", [mask, _mask])
        else:
            mask = _mask

    return table if mask is None else table.filter(mask)


def _pack_columns(
    columns: Sequence[Tensor],
) -> tuple[list[Tensor], str, list[int]]:
    leaves, kinds, metadata = [], [], []
    for column in columns:
        if isinstance(column, StringTensor):
            kinds.append("v" if column._valid is not None else "s")
            leaves.extend((column._data, column._offset))
            if column._valid is not None:
                leaves.append(column._valid)
            metadata.extend(
                (
                    column.ndim,
                    *column.shape,
                    *column.stride(),
                    column._storage_offset,
                )
            )
        elif isinstance(column, NullableTensor):
            kinds.append("n")
            leaves.extend((column._data, column._valid))
        else:
            kinds.append("t")
            leaves.append(column)
    return leaves, "".join(kinds), metadata


@torch.library.custom_op("sdm::_join_indices", mutates_args=())
def _join_indices(
    leaves: list[Tensor],
    kinds: str,
    metadata: list[int],
    left_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    columns = []
    tensor_index = metadata_index = 0
    for kind in kinds:
        if kind in ("s", "v"):
            ndim = metadata[metadata_index]
            size = metadata[metadata_index + 1 : metadata_index + 1 + ndim]
            stride = metadata[
                metadata_index + 1 + ndim : metadata_index + 1 + 2 * ndim
            ]
            storage_offset = metadata[metadata_index + 1 + 2 * ndim]
            has_valid = kind == "v"
            column = StringTensor(
                data=leaves[tensor_index],
                offset=leaves[tensor_index + 1],
                valid=leaves[tensor_index + 2] if has_valid else None,
                size=size,
                stride=stride,
                storage_offset=storage_offset,
            )
            if not column.is_contiguous():
                column = _clone(column, memory_format=torch.contiguous_format)
            tensor_index += 3 if has_valid else 2
            metadata_index += 2 + 2 * ndim
        elif kind == "n":
            column = NullableTensor(
                leaves[tensor_index], leaves[tensor_index + 1]
            )
            tensor_index += 2
        else:
            column = leaves[tensor_index]
            tensor_index += 1
        columns.append(column)
    names = [f"key_{i}" for i in range(left_count)]
    return _join_index_eager(
        TableTensor(
            columns={"id": names},
            id=ColumnarTensor(tuple(columns[:left_count])),
        ),
        TableTensor(
            columns={"id": names},
            id=ColumnarTensor(tuple(columns[left_count:])),
        ),
        names,
        names,
        dtype=dtype,
        device=device,
    )


@_join_indices.register_fake
def _join_indices_fake(
    leaves: list[Tensor],
    kinds: str,
    metadata: list[int],
    left_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    # Both index arrays have the same data-dependent number of matches.
    size = torch.library.get_ctx().new_dynamic_size()
    return (
        torch.empty(size, dtype=dtype, device=device),
        torch.empty(size, dtype=dtype, device=device),
    )
