# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._table import category_values


class RejectNonfinite(PriorTransform):
    """Reject datasets with missing or nonfinite feature or target values."""

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor] | None:
        for table in (x, y):
            values = category_values(table.categorical, torch.float64)
            if not (
                table.numerical.isfinite().all() & values.isfinite().all()
            ):
                return None
        return x, y
