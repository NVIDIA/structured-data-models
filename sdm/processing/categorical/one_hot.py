# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.tensor.categorical import _iter_categories


class OneHotEncode(Processor):
    """Expand categorical vocabularies into dense numerical indicators.

    Each category produces a column named ``<column>__one_hot_<code>`` in
    vocabulary order. Negative codes (missing or unknown values) produce
    all-zero indicators. Indicators use the table's numerical dtype.

    Use :class:`~sdm.processing.AlignCategories` before this processor to
    learn vocabularies from context rows and keep columns consistent for
    independently tensorized inputs.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        codes = table.categorical.code
        sizes = [
            categories.numel()
            for categories in _iter_categories(table.categorical)
        ]
        # Compare each output indicator against its source column and code.
        column_index = codes.new_tensor(
            [index for index, size in enumerate(sizes) for _ in range(size)],
            dtype=torch.long,
        )
        category_index = codes.new_tensor(
            [index for size in sizes for index in range(size)],
        )
        indicators = (
            codes.index_select(-1, column_index) == category_index
        ).to(table.numerical.dtype)
        columns = tuple(
            f"{column}__one_hot_{index}"
            for column, size in zip(table.categorical_columns, sizes)
            for index in range(size)
        )
        out = TableTensor(
            columns={Stype.numerical: (*table.numerical_columns, *columns)},
            numerical=torch.cat((table.numerical, indicators), dim=-1),
        )
        return cast(
            TableTensor,
            torch.cat(
                (table.drop_stypes((Stype.numerical, Stype.categorical)), out),
                dim=-1,
            ),
        )
