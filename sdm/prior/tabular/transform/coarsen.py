# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior import (
    HP,
    Correlated,
    CorrelatedContext,
    LogUniform,
    LogUniformInt,
    Uniform,
)
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import _standardize
from sdm.prior.tabular.transform._table import (
    _unique_values,
    block_order,
    feature_values,
)


class Coarsen(PriorTransform):
    """Reduce selected feature columns to a few ordered levels.

    Args:
        columns: Fraction of eligible columns to coarsen
            per dataset. A value above one selects all columns. Defaults to
            ``Uniform(0.3, 1.3)``.
        num_levels: Sampled number of output levels per column. The columns
            of one dataset share a correlated distribution. Defaults to
            ``LogUniformInt(2, 8)``.
        step: Sampled regular-grid spacing per column. The columns of one
            dataset share a correlated distribution. Defaults to
            ``LogUniform(0.25, 2.0)``.
        grid_probability: Probability of using a regular grid.
        categorical: Whether categorical columns are eligible.
        standardize: Whether to standardize coarsened values.
    """

    def __init__(
        self,
        *,
        columns: HP[float] | float | None = None,
        num_levels: HP[int] | None = None,
        step: HP[float] | None = None,
        grid_probability: float = 0.5,
        categorical: bool = False,
        standardize: bool = False,
    ) -> None:
        self.columns = HP.from_value(
            Uniform(0.3, 1.3) if columns is None else columns
        )
        self.num_levels = Correlated(
            LogUniformInt(2, 8) if num_levels is None else num_levels
        )
        self.step = Correlated(LogUniform(0.25, 2.0) if step is None else step)
        self._grid = Correlated(Uniform(0.0, 1.0))
        self.grid_probability = grid_probability
        self.categorical = categorical
        self.standardize = standardize

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = x.numerical
        if self.categorical:
            values = feature_values(x)
        if values.size(-1) == 0 or values.size(-2) < 2:
            return x, y
        num_rows, num_cols = values.shape
        device = values.device
        with CorrelatedContext(generator=generator):
            frac = self.columns.sample_batch(size=(), generator=generator)
            count = (frac.clamp(max=1.0) * num_cols).round().clamp(min=1)
            col_rank = torch.argsort(
                torch.rand(num_cols, device=device, generator=generator)
            )
            selected = col_rank < count
            bins = self.num_levels.sample_batch(size=num_cols)
            grid = self._grid.sample_batch() < self.grid_probability
            step = self.step.sample_batch(size=num_cols).to(values.dtype)
        num_bins = int(bins.max())

        # Equal-frequency bins by rank, each replaced by its mean:
        noise = torch.randn(
            size=(num_cols, num_rows),
            dtype=torch.float64,
            device=device,
            generator=generator,
        )
        order = (values.mT.double() + 1e-6 * noise).argsort(dim=-1)
        rank = order.argsort(dim=-1).mT
        bin_index = rank * bins // num_rows
        flat = bin_index + num_bins * torch.arange(num_cols, device=device)
        flat = flat.flatten()
        cells = values.flatten()
        sums = torch.zeros(
            num_bins * num_cols, dtype=values.dtype, device=device
        ).scatter_add_(dim=0, index=flat, src=cells)
        counts = torch.zeros_like(sums).scatter_add_(
            dim=0, index=flat, src=torch.ones_like(cells)
        )
        quantile = (sums / counts.clamp(min=1)).gather(0, flat).view_as(values)

        # Round to a step and censor to ``bins`` consecutive integers. Adding
        # zero turns ``-0.0`` into ``0.0``, so zero gives one category:
        low = -(bins // 2)
        grid_values = (values / step).round()
        grid_values = torch.maximum(grid_values, low)
        grid_values = torch.minimum(grid_values, low + bins - 1) + 0.0

        # Censoring can leave one value; that column keeps its quantiles.
        single = (grid_values == grid_values[:1]).all(dim=0)
        coarse = torch.where(grid & ~single, grid_values, quantile)
        if self.standardize:
            coarse = _standardize(coarse)
        apply = selected & values.isfinite().all(dim=0)
        output = torch.where(apply, coarse, values)
        if self.categorical:
            order = block_order(x)
            output = output[:, order]
            values = values[:, order]
        num_numerical = x.numerical.size(-1)
        if not self.categorical or x.categorical.size(-1) == 0:
            return x.replace_blocks(numerical=output), y
        changed = ((output != values) & values.isfinite()).any(dim=0)
        codes = x.categorical.code.clone()
        categories = list(x.categorical.categories)
        for index, selected in enumerate(changed[num_numerical:].tolist()):
            if selected:
                categories[index], code = _unique_values(
                    output[:, num_numerical + index]
                )
                codes[..., index] = code
        return x.replace_blocks(
            numerical=output[:, :num_numerical],
            categorical=CategoricalTensor(
                code=codes, categories=tuple(categories)
            ),
        ), y
