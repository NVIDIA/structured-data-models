# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import (
    _clip_outliers,
    _standardize,
)
from sdm.prior.tabular.transform._table import block_order, feature_values


class ScaleFeatures(PriorTransform):
    """Clip the outliers of each feature column and standardize it.

    A categorical column keeps its codes and gets scaled category values.
    A category without a cell becomes NaN.
    """

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        # The statistics round by column position.
        values = feature_values(x)
        values = _standardize(_clip_outliers(values))[:, block_order(x)]
        num_numerical = x.numerical.size(-1)
        # Each column owns a slot for code -1, then one slot per category.
        code = x.categorical.code
        sizes = [len(category) + 1 for category in x.categorical.categories]
        start = torch.tensor(
            [*accumulate(sizes, initial=1)][:-1],
            dtype=torch.long,
            device=code.device,
        )
        flat = values.new_full(size=(sum(sizes),), fill_value=torch.nan)
        flat[code + start] = values[:, num_numerical:]
        categorical = CategoricalTensor(
            code=code,
            categories=tuple(category[1:] for category in flat.split(sizes)),
        )
        return x.replace_blocks(
            numerical=values[:, :num_numerical], categorical=categorical
        ), y
