# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class MinMaxScale(Processor, InvertibleMixin):
    """Scale each feature column by its minimum and range.

    Values within the fitted range map to ``[0, 1]``; values outside it are
    not clipped. Constant columns use a unit scale to keep the transform
    finite and invertible. NaN and infinite values are ignored when fitting
    statistics and preserved during the transform. Columns without finite
    fitted values transform to NaN.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    minimum: Tensor
    scale: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("minimum", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        nonfinite = ~_isfinite(numerical)
        lower = numerical.masked_fill(nonfinite, math.inf)
        lower = lower.amin(dim=-2, keepdim=True)
        upper = numerical.masked_fill(nonfinite, -math.inf)
        upper = upper.amax(dim=-2, keepdim=True)
        self._set_buffer("minimum", lower)
        scale = torch.where(lower == upper, 1.0, upper - lower)
        self._set_buffer("scale", scale)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical - self.minimum).div_(self.scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical * self.scale).add_(self.minimum)
        return table.replace_blocks(numerical=numerical)
