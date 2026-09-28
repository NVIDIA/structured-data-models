# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.numerical._stats import _isfinite


class ClipSoft(Processor):
    r"""Soft-clip numerical values toward ``±max_absolute_value``.

    Values pass through

    .. math::

        \frac{x}{\sqrt{1 + (x / b)^{2}}}

    which maps monotonically onto ``±max_absolute_value`` (``b``) without a
    hard edge. NaN is preserved. ``±Inf`` maps to ``±max_absolute_value``.
    Unlike :class:`~sdm.processing.ClipSigma`, the bound is fixed rather
    than fitted from per-column statistics.

    Args:
        max_absolute_value: Bound of the soft clip.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, max_absolute_value: float) -> None:
        super().__init__()
        if not math.isfinite(max_absolute_value) or max_absolute_value <= 0:
            raise ValueError("max_absolute_value must be finite and positive.")
        self.max_absolute_value = max_absolute_value

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        bound = self.max_absolute_value
        if torch.is_grad_enabled() and numerical.requires_grad:
            ratio = (numerical / bound).abs()
            squared = 1 + ratio.square()
            unit = torch.where(squared.isfinite(), ratio / squared.sqrt(), 1.0)
            clipped = numerical.sign() * bound * unit
            clipped = torch.where(numerical.isnan(), numerical, clipped)
            return table.replace_blocks(numerical=clipped)

        unit = numerical.div(bound).abs_()
        root = unit.square().add_(1).sqrt_()
        # 'root' is finite exactly where '1 + unit ** 2' is.
        unit.div_(root).masked_fill_(~_isfinite(root), 1.0)
        del root
        clipped = numerical.sign().mul_(bound).mul_(unit)
        torch.where(numerical.isnan(), numerical, clipped, out=clipped)
        return table.replace_blocks(numerical=clipped)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"({self.max_absolute_value})"
        )
