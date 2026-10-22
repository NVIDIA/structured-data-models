# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _isfinite


class Clip(Processor, InvertibleMixin):
    r"""Clip numerical values to a fixed interval.

    With ``method="hard"``, values are clamped to the interval. With
    ``method="saturate"``, values are mapped smoothly into the interval:

    .. math::

        y = \frac{x}{\sqrt{1 + (x / b)^2}}

    where :math:`b` is ``max_value`` and ``min_value`` must equal
    ``-max_value``. NaN is preserved and infinities map to the endpoints.
    Inverse transformation leaves values unchanged.

    Args:
        min_value: Inclusive lower bound for every numerical value.
        max_value: Inclusive upper bound for every numerical value.
        method: ``"hard"`` clamps to the bounds; ``"saturate"`` approaches
            finite, nonzero symmetric bounds smoothly and also changes
            interior values.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def __init__(
        self,
        min_value: float,
        max_value: float,
        *,
        method: Literal["saturate", "hard"] = "hard",
    ) -> None:
        super().__init__()
        if min_value > max_value:
            raise ValueError(
                f"Expected 'min_value' to be less than or equal to "
                f"'max_value' (got {min_value} and {max_value})"
            )
        if method == "saturate" and (
            min_value != -max_value
            or not math.isfinite(max_value)
            or max_value == 0
        ):
            raise ValueError(
                f"Expected finite, nonzero, symmetric bounds for "
                f"method='saturate' (got min_value={min_value}, "
                f"max_value={max_value})"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.method = method

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        if self.method == "hard":
            return table.replace_blocks(
                numerical=numerical.clamp(self.min_value, self.max_value)
            )
        assert self.method == "saturate"
        assert not numerical.requires_grad
        bound = self.max_value
        unit = numerical.div(bound).abs_()
        root = unit.square().add_(1).sqrt_()
        unit.div_(root).masked_fill_(~_isfinite(root), 1.0)
        del root
        clipped = unit.mul_(bound).mul_(numerical.sign())
        torch.where(numerical.isnan(), numerical, clipped, out=clipped)
        return table.replace_blocks(numerical=clipped)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table

    def __repr__(self, *, indent: int = 0) -> str:
        extra = "" if self.method == "hard" else f", method={self.method!r}"
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"{self.min_value}, {self.max_value}{extra})"
        )
