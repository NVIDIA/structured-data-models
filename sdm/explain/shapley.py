# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.explain._tabular import _to_table
from sdm.models import ICLModel

if TYPE_CHECKING:
    from shapiq import InteractionValues


class ShapleyExplainer:
    r"""Compute Shapley values of feature columns from the `"A Unified
    Approach to Interpreting Model Predictions"
    <https://arxiv.org/abs/1705.07874>`_ paper.

    A fitted model is evaluated on subsets of the columns of each query row,
    and the columns outside a subset are removed by imputation. A
    ``shapiq.TabularExplainer`` then distributes the prediction of each row
    over its numerical and then categorical columns.

    Args:
        output: The name of the predicted column to explain, *e.g.*, a class
            or a quantile.
        budget: The number of column subsets evaluated per query row. Values
            are exact from ``2 ** D`` on, for ``D`` columns.
        imputer: How the columns outside a subset are removed. ``"baseline"``
            replaces them by the mean or most frequent category of the
            reference rows, ``"marginal"`` by the values of sampled reference
            rows, and ``"missing"`` by missing values, which the model handles
            itself.
        index: The interaction index of ``shapiq``. If ``None``, Shapley
            values for ``max_order=1`` and k-Shapley interaction values
            (``"k-SII"``) otherwise.
        max_order: The maximum interaction order.
        kwargs: Additional keyword arguments passed to
            ``shapiq.TabularExplainer``, *e.g.*, the ``sample_size`` of
            ``"marginal"``.
    """

    def __init__(
        self,
        output: str,
        budget: int,
        *,
        imputer: Literal["baseline", "marginal", "missing"] = "baseline",
        index: str | None = None,
        max_order: int = 1,
        **kwargs: Any,
    ) -> None:
        self.output = output
        self.budget = budget
        self.imputer = imputer
        self.index = index
        self.max_order = max_order
        self.kwargs = kwargs

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
        *,
        background: TableTensor | None = None,
        generator: torch.Generator | None = None,
    ) -> list[InteractionValues]:
        r"""Explain the predictions of query examples.

        Args:
            model: The fitted model.
            x_query: The feature table of query examples with shape
                ``[R, D]`` with ``R`` rows and ``D`` columns.
            background: The feature table of reference rows with shape
                ``[R_background, D]``, *e.g.*, the in-context examples.
                Required unless ``imputer="missing"``. Categories of the
                query rows that do not occur in it are treated as missing.
            generator: Pseudorandom number generator used for sampling column
                subsets and reference rows.

        Returns:
            The ``shapiq.InteractionValues`` of each row.
        """
        from shapiq import TabularExplainer

        if self.imputer == "missing":
            data = np.full((1, x_query.size(-1)), np.nan)
        else:
            assert background is not None
            if not background.is_same_schema(x_query):
                raise ValueError(
                    "Expected 'background' and 'x_query' to share the same "
                    "schema"
                )
            # Give each category the same code in both tables:
            align = sp.AlignCategories().fit(background)
            background = align.transform(background)
            x_query = align.transform(x_query)
            data = sp.ToNumerical().transform(background).numerical
            if self.imputer == "baseline":  # Let the mean skip missing values:
                data = data.where(~data.isnan(), data.nanmean(dim=0))
            data = data.cpu().double().numpy()
        # shapiq leaves the base value of a single row at 0:
        if len(data) == 1:
            data = data.repeat(2, axis=0)

        def predict(values: np.ndarray) -> np.ndarray:  # [N, D] -> [N]
            # shapiq passes object arrays for "baseline", so cast them:
            values = values.astype(np.float64)
            tensor = torch.as_tensor(values, device=x_query.device)
            pred = model.predict(_to_table(tensor, like=x_query))[self.output]
            return pred.numerical.squeeze(-1).cpu().double().numpy()

        explainer = TabularExplainer(
            model=predict,
            data=data,
            imputer="baseline" if self.imputer == "missing" else self.imputer,
            categorical_features=list(
                range(x_query.numerical.size(-1), x_query.size(-1))
            ),
            index=_index(self.index, self.max_order),
            max_order=self.max_order,
            random_state=_seed(generator),
            **self.kwargs,
        )
        values = sp.ToNumerical().transform(x_query).numerical
        values = values.cpu().double().numpy()
        return list(explainer.explain_X(values, budget=self.budget))


def _seed(generator: torch.Generator | None) -> int:
    # Draw a seed for `shapiq`, whose sampling takes an integer.
    device = None if generator is None else generator.device
    seed = torch.randint(2**31, size=(), generator=generator, device=device)
    return int(seed)


def _index(index: str | None, max_order: int) -> Any:
    # Return the interaction index, which `shapiq` types as a literal.
    if index is not None:
        return index
    return "SV" if max_order == 1 else "k-SII"
