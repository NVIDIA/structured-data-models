# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class RobustScale(Processor, InvertibleMixin):
    """Center and scale each feature column with median and quantile range.

    If the lower and upper quantiles coincide, the scale is half the column
    span. Constant columns become 0. NaN and infinite values are ignored
    when fitting statistics and preserved during the transform.

    Args:
        quantile_range: Percentile pair ``(low, high)`` in ``[0, 100]``
            used as the scale.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__()
        low, high = quantile_range
        if not 0 <= low <= high <= 100:
            raise ValueError(
                "quantile_range must satisfy 0 <= low <= high <= 100."
            )
        self.quantile_range = (low, high)
        self.register_buffer("median", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer("constant", torch.empty(0, dtype=torch.bool))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        finite = numerical.isfinite()
        finite_or_nan = numerical.masked_fill(~finite, torch.nan)
        column_min = torch.where(finite, numerical, torch.inf).amin(
            dim=-2,
            keepdim=True,
        )
        column_max = torch.where(finite, numerical, -torch.inf).amax(
            dim=-2,
            keepdim=True,
        )
        q_low, q_high = (value / 100.0 for value in self.quantile_range)
        lower, self.median, upper = finite_or_nan.nanquantile(
            numerical.new_tensor([q_low, 0.5, q_high]),
            dim=-2,
            keepdim=True,
        )
        self.constant = column_max == column_min
        tiny = torch.finfo(numerical.dtype).tiny
        scale = torch.where(
            lower == upper,
            (column_max - column_min + tiny) / 2,
            upper - lower,
        )
        self.scale = torch.where(self.constant, 1.0, scale)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        scaled = (numerical - self.median) / self.scale
        scaled = torch.where(self.constant & numerical.isfinite(), 0.0, scaled)
        return table.replace_blocks(numerical=scaled)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical * self.scale + self.median
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"quantile_range={self.quantile_range})"
        )
