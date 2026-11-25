# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Sequence

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain._tabular import _to_table
from sdm.models import ICLModel
from sdm.processing import ToNumerical


class ICEExplainer:
    r"""Compute individual conditional expectation (ICE) curves of feature
    columns from the `"Peeking Inside the Black Box: Visualizing Statistical
    Learning with Plots of Individual Conditional Expectation"
    <https://arxiv.org/abs/1309.6392>`_ paper.

    Each column is in turn set to every value of a grid in all query rows of
    a fitted model, which gives one curve of predictions per query row.
    Numerical grids are evenly spaced between the 5% and 95% quantiles of the
    column, and categorical grids hold its observed categories.

    Args:
        output: The name of the predicted column to explain, *e.g.*, a class
            or a quantile.
        features: The name or names of the columns to explain. If ``None``,
            all numerical and then categorical columns.
        grid_resolution: The number of grid values ``G`` of numerical columns.
    """

    def __init__(
        self,
        output: str,
        features: str | Sequence[str] | None = None,
        grid_resolution: int = 20,
    ) -> None:
        self.output = output
        self.features = [features] if isinstance(features, str) else features
        self.grid_resolution = grid_resolution

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
    ) -> dict[str, tuple[TableTensor, Tensor]]:
        r"""Explain the predictions of query examples.

        Args:
            model: The fitted model.
            x_query: The feature table of query examples with shape
                ``[R, D]`` with ``R`` rows and ``D`` columns.

        Returns:
            A dictionary from each column name to its grid table with shape
            ``[G, 1]`` and its curves with shape ``[G, R]``.
        """
        # Categories as codes:
        inputs = ToNumerical().transform(x_query).numerical
        quantiles = inputs.new_tensor([0.05, 0.95])
        steps = torch.linspace(
            start=0.0,
            end=1.0,
            steps=self.grid_resolution,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        names = (*x_query.numerical_columns, *x_query.categorical_columns)
        out: dict[str, tuple[TableTensor, Tensor]] = {}
        for name in names if self.features is None else self.features:
            j = names.index(name)
            column = inputs[:, j]
            if j < x_query.numerical.size(-1):
                low, high = column[column.isfinite()].quantile(quantiles)
                grid = low + (high - low) * steps
            else:
                grid = column[column >= 0].unique()
            values = inputs.repeat(len(grid), 1)  # [G * R, D]
            values[:, j] = grid.repeat_interleave(len(x_query))
            table = _to_table(values, like=x_query)
            pred = model.predict(table)[self.output].numerical  # [G * R, 1]
            curves = pred.view(len(grid), len(x_query))  # [G, R]
            out[name] = (table[name][:: len(x_query)], curves)
        return out


class PartialDependenceExplainer(ICEExplainer):
    r"""Compute the partial dependence of feature columns from the `"Greedy
    Function Approximation: A Gradient Boosting Machine"
    <https://doi.org/10.1214/aos/1013203451>`_ paper.

    The partial dependence of a column is the mean of its curves of
    :class:`~sdm.explain.ICEExplainer` over all query rows.

    Args:
        output: The name of the predicted column to explain, *e.g.*, a class
            or a quantile.
        features: The name or names of the columns to explain. If ``None``,
            all numerical and then categorical columns.
        grid_resolution: The number of grid values ``G`` of numerical columns.
    """

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
    ) -> dict[str, tuple[TableTensor, Tensor]]:
        r"""Explain the mean prediction of query examples.

        Args:
            model: The fitted model.
            x_query: The feature table of query examples with shape
                ``[R, D]`` with ``R`` rows and ``D`` columns.

        Returns:
            A dictionary from each column name to its grid table with shape
            ``[G, 1]`` and its partial dependence with shape ``[G]``.
        """
        curves = super().explain(model, x_query)
        return {
            name: (grid, pred.mean(dim=-1))  # [G, R] -> [G]
            for name, (grid, pred) in curves.items()
        }
