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
    ``method="saturate"``, values are mapped smoothly into it:

    .. math::

        y = c + \frac{h z}{\sqrt{1 + z^2}}, \quad z = \frac{x - c}{h},

    where :math:`c` and :math:`h` are the midpoint and half-width of
    ``[min_value, max_value]``. NaN is preserved and infinities map to the
    endpoints. Inverse transformation leaves values unchanged.

    Args:
        min_value: Inclusive lower bound for every numerical value.
        max_value: Inclusive upper bound for every numerical value.
        method: ``"hard"`` clamps to the bounds; ``"saturate"`` approaches
            finite, distinct bounds smoothly and also changes interior values.
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
                "min_value must be less than or equal to max_value"
            )
        if method == "saturate" and (
            not math.isfinite(min_value)
            or not math.isfinite(max_value)
            or min_value == max_value
        ):
            raise ValueError("Saturation requires finite, distinct bounds.")
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
        center = self.min_value / 2 + self.max_value / 2
        bound = self.max_value / 2 - self.min_value / 2
        centered = numerical if center == 0 else numerical - center
        unit = centered.div(bound).abs_()
        root = unit.square().add_(1).sqrt_()
        unit.div_(root).masked_fill_(~_isfinite(root), 1.0)
        del root
        clipped = unit.mul_(bound).mul_(centered.sign())
        if center != 0:
            clipped.add_(center)
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
