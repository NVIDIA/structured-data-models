# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class RobustScale(Processor, InvertibleMixin):
    """Center and scale each feature column with median and quantile range.

    Columns whose lower and upper quantile coincide use a unit scale to keep
    the transform finite and invertible. NaN and infinite values are ignored
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

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        finite_or_nan = numerical.masked_fill(~numerical.isfinite(), torch.nan)
        q_low, q_high = (value / 100.0 for value in self.quantile_range)
        # 'nanquantile' requires single or double precision input.
        quantile_input = finite_or_nan.to(
            dtype=torch.promote_types(numerical.dtype, torch.float32),
        )
        lower, median, upper = quantile_input.nanquantile(
            quantile_input.new_tensor([q_low, 0.5, q_high]),
            dim=-2,
            keepdim=True,
        )
        self.median = median.to(dtype=numerical.dtype)
        scale = torch.where(lower == upper, 1.0, upper - lower)
        self.scale = scale.to(dtype=numerical.dtype)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = (table.numerical - self.median) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical * self.scale + self.median
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"quantile_range={self.quantile_range})"
        )
