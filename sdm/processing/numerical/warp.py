# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class WarpTails(Processor):
    r"""Warp numerical columns to more skew or heavier tails.

    When the processor is fitted, each column is selected with
    ``probability``. A selected column gets one of two maps:
    :math:`x^{a}` for :math:`x \geq 0` and :math:`-|x|^{b}` otherwise, which
    adds skew; or :math:`\mathrm{sign}(x) (e^{s |x|} - 1)`, which adds heavy
    tails. Standardize the columns before this step.

    Args:
        probability: Probability that a column is warped.
        exponent: Range of the exponents :math:`a` and :math:`b`.
        scale: Range of the scale :math:`s`.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        probability: float = 0.4,
        exponent: tuple[float, float] = (0.05, 5.0),
        scale: tuple[float, float] = (0.3, 2.0),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.exponent = exponent
        self.scale = scale
        self.register_buffer("column_exponent", torch.empty(0))
        self.register_buffer("column_scale", torch.empty(0))
        self.register_buffer("power", torch.empty(0, dtype=torch.bool))
        self.register_buffer("exp", torch.empty(0, dtype=torch.bool))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        *batch, _, num_columns = table.numerical.shape
        device = table.device
        low, high = self.exponent
        self.column_exponent = (
            torch.empty(
                size=(*batch, 2, 1, num_columns),
                device=device,
                dtype=table.numerical.dtype,
            )
            .uniform_(math.log(low), math.log(high), generator=generator)
            .exp_()
        )
        low, high = self.scale
        self.column_scale = (
            torch.empty(
                size=(*batch, 1, num_columns),
                device=device,
                dtype=table.numerical.dtype,
            )
            .uniform_(math.log(low), math.log(high), generator=generator)
            .exp_()
        )
        u = torch.rand(
            (2, *batch, 1, num_columns), device=device, generator=generator
        )
        selected = u[0] < self.probability
        self.power = selected & (u[1] < 0.5)
        self.exp = selected & (u[1] >= 0.5)

    def _transform(self, table: TableTensor) -> TableTensor:
        x = table.numerical
        magnitude = x.abs()
        a, b = self.column_exponent.unbind(-3)
        skewed = torch.where(x >= 0, magnitude.pow(a), -magnitude.pow(b))
        heavy = x.sign() * (magnitude * self.column_scale).expm1()
        numerical = torch.where(
            self.power, skewed, torch.where(self.exp, heavy, x)
        )
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"probability={self.probability}, exponent={self.exponent}, "
            f"scale={self.scale})"
        )
