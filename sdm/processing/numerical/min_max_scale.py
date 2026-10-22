# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class MinMaxScale(Processor, InvertibleMixin):
    """Scale each numerical column by its fitted min-max range.

    NaN and infinite values are ignored when fitting statistics and preserved
    during the transform.
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
        x = table.numerical
        missing = ~_isfinite(x)
        minimum = x.masked_fill(missing, torch.inf).amin(-2, keepdim=True)
        minimum[minimum.isinf()] = torch.nan
        maximum = x.masked_fill(missing, -torch.inf).amax(-2, keepdim=True)
        scale = maximum - minimum
        scale[scale == 0] = 1
        self._set_buffer("minimum", minimum)
        self._set_buffer("scale", scale)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical - self.minimum).div_(self.scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical * self.scale).add_(self.minimum)
        return table.replace_blocks(numerical=numerical)
