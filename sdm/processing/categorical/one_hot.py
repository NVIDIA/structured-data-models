# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.tensor.categorical import _iter_categories


class OneHotEncode(Processor):
    """One-hot encode categorical columns as dense numerical columns.

    Each category produces a column named ``<column>_<code>`` in
    vocabulary order. Missing and unknown values encode as zeros.

    Use :class:`~sdm.processing.AlignCategories` first to align context and
    query vocabularies.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        codes = table.categorical.code
        sizes = [
            categories.numel()
            for categories in _iter_categories(table.categorical)
        ]
        columns = tuple(
            f"{column}_{index}"
            for column, size in zip(table.categorical_columns, sizes)
            for index in range(size)
        )
        numerical = table.numerical
        if columns:
            num_numerical = numerical.size(-1)
            out = numerical.new_zeros(
                (*codes.shape[:-1], num_numerical + len(columns))
            )
            if num_numerical:
                out[..., :num_numerical] = numerical
            offsets = codes.new_tensor(
                list(accumulate(sizes, initial=num_numerical))[:-1],
                dtype=torch.long,
            )
            known = codes >= 0
            indices = torch.where(known, codes + offsets, num_numerical)
            out.scatter_add_(-1, indices, known.to(numerical.dtype))
            numerical = out
        return TableTensor(
            columns={
                **table.columns,
                Stype.numerical: (*table.numerical_columns, *columns),
                Stype.categorical: (),
            },
            numerical=numerical,
            **{
                stype.value: getattr(table, stype.value)
                for stype in table.columns
                if stype not in (Stype.numerical, Stype.categorical)
            },
        )
