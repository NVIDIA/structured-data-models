# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP, Correlated
from sdm.prior.converters import NumericalConverter


class KumaraswamyNumericalConverter(NumericalConverter):
    """Apply Kumaraswamy warping after min-max scaling each column.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
    """

    def __init__(
        self,
        a: HP[float] | Correlated[float] | float,
        b: HP[float] | Correlated[float] | float,
    ) -> None:
        self.a = HP.from_value(a)
        self.b = HP.from_value(b)

    def forward(  # noqa: D102
        self,
        slices: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = slices.values() if slices.is_nested else slices.squeeze(1)
        num_cols = x.size(0)

        a = (
            self.a.sample_batch(
                size=num_cols,
                generator=generator,
            )
            .to(x.dtype)
            .unsqueeze(-1)
        )
        b = (
            self.b.sample_batch(
                size=num_cols,
                generator=generator,
            )
            .to(x.dtype)
            .unsqueeze(-1)
        )

        # Min-max scale to `[0, 1]`:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        x = ((x - x_min) / (x_max - x_min + 1e-30)).clamp(0.0, 1.0)

        return 1 - (1 - x.pow(a)).pow(b), slices
