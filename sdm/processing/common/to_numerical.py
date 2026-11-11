# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from sdm import EnsembleData, Stype, TableLike, TableTensor
from sdm.processing import Processor


class ToNumerical(Processor):
    """Move categorical codes into the numerical block.

    Existing numerical columns come first; categorical codes are cast to
    their dtype. Missing categorical values remain ``-1``.
    """

    requires_fit = False
    handles_stypes = (Stype.numerical, Stype.categorical)

    def _transform(self, table: TableLike) -> TableLike:
        return EnsembleData.execute(self._convert, table, batched=True)

    def _convert(self, table: TableTensor) -> TableTensor:
        if table.categorical.size(-1) == 0:
            return table

        # Casting to the (floating-point) numerical dtype also unwraps a
        # CategoricalTensor to its raw ordinal ids as a plain tensor.
        categorical = table.categorical.to(table.numerical.dtype)
        columns = (
            *table.numerical_columns,
            *table.categorical_columns,
        )
        numerical = (
            categorical
            if table.numerical.size(-1) == 0
            else torch.cat((table.numerical, categorical), dim=-1)
        )
        out = table.__class__(
            columns={Stype.numerical: columns},
            numerical=numerical,
        )
        return cast(
            TableTensor,
            torch.cat(
                (table.drop_stypes((Stype.numerical, Stype.categorical)), out),
                dim=-1,
            ),
        )
