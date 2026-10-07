# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.numerical._stats import _isfinite, _nanmean_rows


class ImputeMean(Processor):
    """Replace NaN feature values with fitted per-column means.

    Infinite values are ignored when fitting the mean but preserved during
    transformation.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (columns without finite values).
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

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
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        mean = _nanmean_rows(
            numerical.masked_fill(~_isfinite(numerical), torch.nan)
        )
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace NaNs with the fitted per-column means."""
        numerical = table.numerical
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        return table.replace_blocks(numerical=numerical)
