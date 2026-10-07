# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _isfinite,
    _nansum_rows,
)


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible. NaN and infinite values are ignored when fitting statistics
    and preserved during the transform.

    Args:
        eps: Value added to each fitted standard deviation.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, *, eps: float = 0.0) -> None:
        super().__init__()
        if eps < 0:
            raise ValueError("epsilon must be non-negative.")
        self.eps = eps
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        self._fit_tensor(table.numerical)

    def _fit_tensor(self, numerical: Tensor) -> None:
        finite = _isfinite(numerical)
        count = finite.sum(dim=-2, keepdim=True)
        finite_or_nan = numerical.masked_fill(~finite, torch.nan)

        self.mean = _nansum_rows(finite_or_nan).div_(count)
        self.mean.masked_fill_(self.mean.isnan(), 0.0)

        var = _nansum_rows(finite_or_nan.sub_(self.mean).square_())
        var /= count
        var.masked_fill_(var.isnan(), 0.0)

        self.scale = var.sqrt()
        if self.eps == 0:
            mask = _constant_feature_mask(var, self.mean, num_samples=count)
            self.scale[mask] = 1.0
        else:
            self.scale += self.eps

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=self._transform_tensor(table.numerical)
        )

    def _transform_tensor(self, numerical: Tensor) -> Tensor:
        return (numerical - self.mean).div_(self.scale)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        dtype = table.numerical.dtype
        numerical = (table.numerical * self.scale).add_(self.mean)
        return table.replace_blocks(numerical=numerical.to(dtype=dtype))

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(eps={self.eps})"
