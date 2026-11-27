# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, NaT, Stype, TableTensor
from sdm.models import ICLModel


class ICEExplainer:
    r"""Compute individual conditional expectation (ICE) curves of feature
    columns from the `"Peeking Inside the Black Box: Visualizing Statistical
    Learning with Plots of Individual Conditional Expectation"
    <https://arxiv.org/abs/1309.6392>`_ paper.

    Each column is in turn set to every value of a grid in all query rows of
    a fitted model, which gives one curve of predictions per query row.
    Numerical and datetime grids are evenly spaced between the 5% and 95%
    quantiles of the column, and categorical grids hold its vocabulary.

    Args:
        grid_resolution: The number of grid values ``G`` of numerical columns.
    """

    def __init__(self, grid_resolution: int = 20) -> None:
        self.grid_resolution = grid_resolution

    def explain(
        self,
        model: ICLModel,
        x: TableTensor,
        output: str,
        inputs: str | Iterable[str] | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        r"""Explain how feature values affect predictions.

        Args:
            model: The fitted model.
            x: The feature tensor of query examples with shape ``[..., R, D]``
                with ``R`` rows and ``D`` columns.
            output: The name of the predicted column to explain.
            inputs: The name or names of the input feature columns to explain.
                If ``None``, all supported columns will be used.

        Returns:
            A dictionary from each column name to its grid table with shape
            ``[..., G]`` and its curves with shape ``[..., R, G]``.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = None if inputs is None else set(inputs)

        result: dict[str, tuple[Tensor, Tensor]] = {}
        for stype, block in x.items():
            if stype not in (
                Stype.numerical,
                Stype.categorical,
                Stype.datetime,
            ):
                continue
            for i, name in enumerate(x.columns[stype]):
                if inputs is not None and name not in inputs:
                    continue

                column = block[..., i : i + 1]

                if stype == Stype.categorical:
                    assert isinstance(column, CategoricalTensor)
                    grid = column.categories[0]
                    grid = grid.expand(*block.size()[:-2], -1)
                    column = CategoricalTensor(
                        code=torch.arange(
                            grid.size(-1),
                            device=x.device,
                            dtype=column.dtype,
                        ).unsqueeze(-1),
                        categories=column.categories,
                    )
                else:
                    column = column.squeeze(-1)
                    if stype == Stype.datetime:
                        mask = column == NaT
                        column = column.double()
                        column[mask] = torch.nan
                    else:
                        column = column.masked_fill(
                            ~column.isfinite(),
                            torch.nan,
                        )

                    low, high = column.nanquantile(
                        column.new_tensor([0.05, 0.95]),
                        dim=-1,
                    )
                    low, high = low.unsqueeze(-1), high.unsqueeze(-1)
                    step = torch.linspace(
                        start=0.0,
                        end=1.0,
                        steps=self.grid_resolution,
                        dtype=column.dtype,
                        device=column.device,
                    )
                    grid = low + (high - low) * step
                    if stype == Stype.datetime:
                        grid = grid.round().to(block.dtype)
                    column = grid.unsqueeze(-1)

                x_i = cast(TableTensor, torch.cat([x] * grid.size(-1), dim=-2))
                before, _, after = getattr(x_i, stype).tensor_split(
                    [i, i + 1],
                    dim=-1,
                )
                column = column.unsqueeze(-2).expand(
                    *x.size()[:-2], grid.size(-1), x.size(-2), 1
                )
                column = column.reshape(
                    *x.size()[:-2], grid.size(-1) * x.size(-2), 1
                )
                x_i = x_i.replace_blocks(
                    **{stype: torch.cat([before, column, after], dim=-1)},
                )
                out = model.predict(x_i)[output].numerical[..., 0]
                curve = out.unflatten(-1, (grid.size(-1), x.size(-2)))
                curve = curve.transpose(-1, -2)
                result[name] = (grid, curve)

        return result


class PartialDependenceExplainer(ICEExplainer):
    r"""Compute the partial dependence of feature columns from the `"Greedy
    Function Approximation: A Gradient Boosting Machine"
    <https://doi.org/10.1214/aos/1013203451>`_ paper.

    The partial dependence of a column is the mean of its curves of
    :class:`~sdm.explain.ICEExplainer` over all query rows.

    Args:
        grid_resolution: The number of grid values ``G`` of numerical columns.
    """

    def explain(
        self,
        model: ICLModel,
        x: TableTensor,
        output: str,
        inputs: str | Iterable[str] | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        r"""Explain how feature values affect predictions.

        Args:
            model: The fitted model.
            x: The feature tensor of query examples with shape ``[..., R, D]``
                with ``R`` rows and ``D`` columns.
            output: The name of the predicted column to explain.
            inputs: The name or names of the input feature columns to explain.
                If ``None``, all supported columns will be used.

        Returns:
            A dictionary from each column name to its grid table with shape
            ``[..., G]`` and its partial dependence with shape ``[..., G]``.
        """
        result = super().explain(model, x, output, inputs)
        return {
            name: (grid, curve.mean(dim=-2))
            for name, (grid, curve) in result.items()
        }
