# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _constant_feature_mask


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
        self.register_buffer("mean", torch.empty(0, dtype=torch.float64))
        self.register_buffer("scale", torch.empty(0, dtype=torch.float64))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        numerical = table.numerical.double()
        finite = numerical.isfinite()
        finite_or_nan = numerical.masked_fill(~finite, torch.nan)

        self.mean = finite_or_nan.nanmean(-2, keepdim=True)
        self.mean.masked_fill_(self.mean.isnan(), 0.0)

        var = (finite_or_nan - self.mean).square().nanmean(-2, keepdim=True)
        var.masked_fill_(var.isnan(), 0.0)

        self.scale = var.sqrt()
        if self.eps == 0:
            mask = _constant_feature_mask(
                var,
                self.mean,
                num_samples=finite.sum(-2, keepdim=True),
            )
            self.scale[mask] = 1.0
        else:
            self.scale += self.eps

    def _transform(self, table: TableTensor) -> TableTensor:
        dtype = table.numerical.dtype
        numerical = table.numerical.double()
        numerical = (numerical - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical.to(dtype=dtype))

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        dtype = table.numerical.dtype
        numerical = table.numerical.double()
        numerical = numerical * self.scale + self.mean
        return table.replace_blocks(numerical=numerical.to(dtype=dtype))

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(eps={self.eps})"
