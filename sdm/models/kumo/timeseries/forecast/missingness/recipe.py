# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed-history preprocessing for missingness-aware forecasting."""

import torch

import sdm.processing as sp
from sdm import TableTensor


class StandardizeObserved(sp.Standardize):
    """Fit Kumo-TS observed-value statistics while preserving missing entries.

    Statistics use float64 accumulation. Empty and nearly constant columns
    use unit standard deviation, and ``1e-8`` is added to every scale.
    Fitted statistics remain fixed during autoregressive forecasting.
    """

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        x = table.numerical.double()
        observed = x.isfinite()
        count = observed.sum(-2, keepdim=True)
        mean = x.masked_fill(~observed, 0).sum(
            -2, keepdim=True
        ) / count.clamp_min(1)
        var = (
            (x - mean).square().masked_fill(~observed, 0).sum(-2, keepdim=True)
        )
        scale = (var / count.clamp_min(1)).sqrt()
        scale = torch.where(scale < 1e-8, 1, scale) + 1e-8
        self._set_buffer("mean", mean)
        self._set_buffer("scale", scale)

    def _transform(self, table: TableTensor) -> TableTensor:
        out = super()._transform(table)
        return out.replace_blocks(
            numerical=out.numerical.to(table.numerical.dtype)
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        out = super()._inverse_transform(table)
        return out.replace_blocks(
            numerical=out.numerical.to(table.numerical.dtype)
        )


def default_recipe() -> sp.Recipe:
    """Normalize observed context values without imputing their gaps."""
    return sp.Recipe(
        features=StandardizeObserved(),
        target=StandardizeObserved(),
        output=sp.AverageEstimators(),
    )
