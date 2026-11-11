# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleData, Stype, TableLike
from sdm.processing import Processor
from sdm.processing.numerical._stats import _isfinite


class ImputeMean(Processor):
    """Replace missing numerical values with fitted per-column means.

    Infinite values are ignored when fitting the mean but preserved during
    transformation.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (columns without finite values).
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _mean: Tensor

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        mean = torch.nanmean(
            numerical.masked_fill(
                ~EnsembleData.execute(_isfinite, numerical, batched=True),
                torch.nan,
            ),
            dim=-2,
            keepdim=True,
        )
        self._set_buffer(
            name="_mean",
            value=torch.where(mean.isnan(), self.fill_value, mean),
        )

    def _transform(self, table: TableLike) -> TableLike:
        numerical = table.numerical
        numerical = EnsembleData.execute(
            torch.where, numerical.isnan(), self._mean, numerical
        )
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        if self.fill_value == 0.0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"fill_value={self.fill_value!r})"
        )
