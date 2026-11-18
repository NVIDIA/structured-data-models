# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior import HP, Uniform
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._table import mask_cells


class MaskRandom(PriorTransform):
    """Independently mask numerical and categorical feature cells.

    Args:
        rate: Masking rate per dataset.
    """

    def __init__(self, rate: HP[float] | float) -> None:
        self.rate = HP.from_value(rate)

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        rate = (
            x.numerical.new_empty(()).uniform_(
                self.rate.low, self.rate.high, generator=generator
            )
            if isinstance(self.rate, Uniform)
            else self.rate.sample(generator=generator)
        )
        num_columns = x.numerical.size(-1) + x.categorical.size(-1)
        missing = (
            torch.rand(
                size=(x.size(-2), num_columns),
                device=x.device,
                generator=generator,
            )
            < rate
        )
        return mask_cells(x, missing), y
