# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import TableTensor


def textify(table: TableTensor) -> tuple[pa.Array, Tensor]:
    """Join cells as ``The {column} is {value}.`` sentences per row.

    Args:
        table: The input table with shape ``[..., C]``, where ``C`` is the
            number of columns.

    Returns:
        A flat :class:`pyarrow.Array` of row strings and a CPU tensor of
        character spans with shape ``[..., C, 2]``.
    """
    dtype = pa.large_string()
    if table.size(-1) == 0:
        array = pa.repeat(pa.scalar("", dtype), math.prod(table.size()[:-1]))
        span = torch.empty((*table.size(), 2), dtype=torch.int64)
        return (array, span)

    _table = table.to_arrow()
    cells, counts = [], []
    for column, array in zip(_table.column_names, _table.itercolumns()):
        if pa.types.is_floating(array.type):
            array = pc.if_else(pc.is_nan(array), None, array)  # type: ignore

        array = array.cast(dtype)
        array = pc.fill_null(array, pa.scalar("missing", dtype))
        cell = pc.binary_join_element_wise(  # type: ignore
            pa.scalar(f"The {column} is ", dtype),
            array,
            pa.scalar(".", dtype),
            pa.scalar("", dtype),
        )
        cells.append(cell)
        counts.append(pc.utf8_length(cell).to_numpy())  # type: ignore

    count = torch.from_numpy(np.stack(counts, axis=-1)).view(table.size())
    end = (count + 1).cumsum(dim=-1) - 1
    span = torch.stack([end - count, end], dim=-1)
    array = pc.binary_join_element_wise(  # type: ignore
        *cells,
        pa.scalar(" ", dtype),
    )
    return array.combine_chunks(), span
