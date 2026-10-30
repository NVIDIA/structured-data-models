# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _isfinite,
)


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible. NaN and infinite values are ignored when fitting statistics
    and preserved during the transform.

    Inverse transformation supports multiple consecutive outputs per fitted
    column, such as quantile predictions.

    Args:
        eps: Value added to each fitted standard deviation.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    mean: Tensor
    scale: Tensor

    def __init__(self, *, eps: float = 0.0) -> None:
        super().__init__()
        if eps < 0:
            raise ValueError("epsilon must be non-negative.")
        self.eps = eps
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def with_state(self, mean: Tensor | float, scale: Tensor | float) -> Self:
        """Set fixed statistics in place and disable refitting.

        Args:
            mean: The mean to subtract.
            scale: The scale to divide by.
        """
        mean = torch.atleast_1d(torch.as_tensor(mean))
        scale = torch.atleast_1d(torch.as_tensor(scale))
        self._set_buffer("mean", mean)
        self._set_buffer("scale", scale)
        self.requires_fit = False
        return self

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        finite = _isfinite(table.numerical)
        count = finite.sum(dim=-2, keepdim=True)
        finite_or_nan = table.numerical.masked_fill(~finite, torch.nan)

        self._set_buffer(
            "mean", finite_or_nan.nansum(-2, keepdim=True).div_(count)
        )
        self.mean.masked_fill_(self.mean.isnan(), 0.0)

        var = finite_or_nan.sub_(self.mean).square_().nansum(-2, keepdim=True)
        var /= count
        var.masked_fill_(var.isnan(), 0.0)

        self._set_buffer("scale", var.sqrt())
        if self.eps == 0:
            mask = _constant_feature_mask(var, self.mean, num_samples=count)
            self.scale[mask] = 1.0
        else:
            self.scale.add_(self.eps)

    def _transform(self, table: TableTensor) -> TableTensor:
        mean = self.mean.to(table.device)
        scale = self.scale.to(table.device)
        numerical = (table.numerical - mean).div_(scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.unflatten(-1, (self.scale.size(-1), -1))
        numerical = numerical * self.scale.to(table.device).unsqueeze(-1)
        numerical.add_(self.mean.to(table.device).unsqueeze(-1))
        return table.replace_blocks(numerical=numerical.flatten(-2))

    def __repr__(self, *, indent: int = 0) -> str:
        if self.eps == 0:
            return super().__repr__(indent=indent)
        return f"{' ' * indent}{self.__class__.__name__}(eps={self.eps})"
