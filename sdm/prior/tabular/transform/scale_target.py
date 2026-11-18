# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import (
    _clip_outliers,
    _standardize,
)


class ScaleTarget(PriorTransform):
    """Clip the outliers of a numerical target and standardize it."""

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        if y.numerical.size(-1) == 0:
            return x, y
        return x, y.replace_blocks(
            numerical=_standardize(_clip_outliers(y.numerical))
        )
