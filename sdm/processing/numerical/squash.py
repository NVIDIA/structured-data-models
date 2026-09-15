# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor


class SquashTransform(Processor):
    r"""Center and scale each feature column robustly, then soft clip it.

    Each column is centered on its median and divided by its quantile range,
    then mapped through :math:`x / \sqrt{1 + (x / b)^2}`, which bounds it to
    ``+-max_absolute_value`` without a hard edge. Columns whose quantiles
    coincide are scaled by their full range instead, and constant columns
    become zero. NaN and infinite values are ignored when fitting statistics;
    NaN values are preserved and infinite values map onto the bounds during
    the transform.

    Args:
        max_absolute_value: Bound ``b`` of the soft clip.
        quantile_range: Lower and upper percentiles in ``[0, 100]`` whose
            difference scales each column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    center: Tensor
    scale: Tensor
    zero_columns: Tensor

    def __init__(
        self,
        *,
        max_absolute_value: float = 3.0,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        super().__init__()
        self.max_absolute_value = max_absolute_value
        self.quantile_range = quantile_range
        self.register_buffer("center", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))
        self.register_buffer(
            "zero_columns",
            torch.empty(0, dtype=torch.bool),
        )

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
        low, high = self.quantile_range
        lower, median, upper = finite_or_nan.nanquantile(
            numerical.new_tensor([low / 100, 0.5, high / 100]),
            dim=-2,
            keepdim=True,
        )

        self.zero_columns = column_max == column_min
        tiny = torch.finfo(numerical.dtype).tiny
        scale = torch.where(
            lower == upper,
            (column_max - column_min + tiny) / 2,
            upper - lower,
        )
        self.center = median
        self.scale = torch.where(self.zero_columns, 1.0, scale)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        out = (numerical - self.center) / self.scale
        out = torch.where(self.zero_columns & ~numerical.isnan(), 0.0, out)
        bound = self.max_absolute_value
        out = out / (1 + (out / bound).square()).sqrt()
        out = torch.where(numerical.isinf(), numerical.sign() * bound, out)
        return table.replace_blocks(numerical=out)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self.max_absolute_value != 3.0:
            arguments.append(f"max_absolute_value={self.max_absolute_value}")
        if self.quantile_range != (25.0, 75.0):
            arguments.append(f"quantile_range={self.quantile_range}")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
