# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._table import feature_order


class ShuffleColumns(PriorTransform):
    """Permute the numerical and categorical feature columns together.

    The column at the new position ``k`` gets the name ``x_{k}``.
    """

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        order = feature_order(x)
        perm = torch.randperm(len(order), device=x.device, generator=generator)
        names = (*x.numerical_columns, *x.categorical_columns)
        rename = {
            names[order[i]]: f"x_{k}" for k, i in enumerate(perm.tolist())
        }
        return TableTensor(
            columns={
                stype.value: tuple(rename.get(name, name) for name in columns)
                for stype, columns in x.columns.items()
            },
            numerical=x.numerical,
            categorical=x.categorical,
            datetime=x.datetime,
            text=x.text,
            id=x.id,
        ), y
