# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate

import torch

from sdm import EnsembleData, EnsembleTable, Stype, TableTensor
from sdm.processing import Processor
from sdm.tensor.categorical import _iter_categories


class ToOneHot(Processor):
    """One-hot encode categorical columns.

    Each category produces a column named ``<column>_<code>`` in
    vocabulary order. Missing and unknown values encode as zeros.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = False

    def _transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        return EnsembleData.execute(self._transform_table, table, batched=True)

    def _transform_table(self, table: TableTensor) -> TableTensor:
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
        # String keys: compiled code cannot run with `Stype` keys.
        schema = {stype.value: names for stype, names in table.columns.items()}
        schema[Stype.numerical.value] = (*table.numerical_columns, *columns)
        schema[Stype.categorical.value] = ()
        return TableTensor(
            columns=schema,
            numerical=numerical,
            **{
                stype.value: getattr(table, stype.value)
                for stype in table.columns
                if stype not in (Stype.numerical, Stype.categorical)
            },
        )
