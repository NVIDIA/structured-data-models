# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class MinMaxScale(Processor, InvertibleMixin):
    """Scale each numerical column by its fitted minimum and range.

    Fitted minima map to zero and maxima to one. Values outside the fitted
    range are not clipped. Constant columns use a unit scale.
    NaN and infinite values are ignored when fitting statistics and preserved
    during the transform. Columns without finite fitting values yield NaN.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _minimum: Tensor
    _scale: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_minimum", torch.empty(0))
        self.register_buffer("_scale", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        x = table.numerical
        missing = ~_isfinite(x)
        minimum = x.masked_fill(missing, torch.inf).amin(-2, keepdim=True)
        maximum = x.masked_fill(missing, -torch.inf).amax(-2, keepdim=True)
        minimum.masked_fill_(minimum.isinf(), torch.nan)
        scale = maximum - minimum
        self._set_buffer("_minimum", minimum)
        self._set_buffer("_scale", scale.masked_fill_(scale == 0, 1))

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical - self._minimum).div_(self._scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical * self._scale).add_(self._minimum)
        return table.replace_blocks(numerical=numerical)
