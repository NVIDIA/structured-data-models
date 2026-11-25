# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from functools import cache, partial
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


class RecontextualizedShapleyExplainer:
    r"""Compute Shapley values of feature columns by recontextualization from
    the `"Interpretable Machine Learning for TabPFN"
    <https://arxiv.org/abs/2403.10923>`_ paper.

    The columns outside a subset are dropped from both the in-context and the
    query examples, so that the model predicts as if it had never seen them.
    Each subset takes one model call for all query rows. A
    ``shapiq.AgnosticExplainer`` then distributes the prediction of each row
    over its numerical and then categorical columns.

    Args:
        output: The name of the predicted column to explain, *e.g.*, a class
            or a quantile.
        budget: The number of column subsets evaluated per query row. Values
            are exact from ``2 ** D`` on, for ``D`` columns.
        index: The interaction index of ``shapiq``. If ``None``, Shapley
            values for ``max_order=1`` and k-Shapley interaction values
            (``"k-SII"``) otherwise.
        max_order: The maximum interaction order.
    """

    def __init__(
        self,
        output: str,
        budget: int,
        *,
        index: str | None = None,
        max_order: int = 1,
    ) -> None:
        self.output = output
        self.budget = budget
        self.index = index
        self.max_order = max_order

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
        *,
        x_context: TableTensor,
        y_context: TableTensor,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> list[InteractionValues]:
        r"""Explain the predictions of query examples.

        Args:
            model: The model. Its fitted state is neither used nor changed,
                so repeat the arguments of its fit to explain the same
                predictions.
            x_query: The feature table of query examples with shape
                ``[R_query, D]`` with ``R_query`` rows and ``D`` columns.
            x_context: The feature table of in-context examples with shape
                ``[R_context, D]``.
            y_context: The targets of in-context examples with shape
                ``[R_context, 1]``.
            generator: Pseudorandom number generator used for sampling column
                subsets, which all query rows share, and for the model calls.
                Both start from its current state and leave it unchanged.
                Without it, each model call samples its ensemble anew.
            kwargs: Additional keyword arguments passed to the model call,
                *e.g.*, ``num_estimators``.

        Returns:
            The ``shapiq.InteractionValues`` of each row.
        """
        from shapiq import AgnosticExplainer

        names = (*x_query.numerical_columns, *x_query.categorical_columns)

        def state() -> torch.Generator | None:  # Leave `generator` unchanged.
            return None if generator is None else generator.clone_state()

        random_state = _seed(state())  # All rows share their column subsets.

        @cache
        def predict(coalition: tuple[bool, ...]) -> np.ndarray:  # [R_query]
            absent = [name for name, keep in zip(names, coalition) if not keep]
            pred = model(
                x_context=x_context.drop_columns(absent),
                y_context=y_context,
                x_query=x_query.drop_columns(absent),
                generator=state(),
                **kwargs,
            )[self.output]
            return pred.numerical.squeeze(-1).cpu().double().numpy()

        def game(coalitions: np.ndarray, row: int) -> np.ndarray:  # [N]
            return np.array(
                [predict(tuple(c))[row] for c in coalitions.tolist()]
            )

        values: list[InteractionValues] = []
        for row in range(len(x_query)):  # shapiq explains one row per game.
            explainer = AgnosticExplainer(
                game=partial(game, row=row),
                n_players=len(names),
                index=_index(self.index, self.max_order),
                max_order=self.max_order,
                random_state=random_state,
            )
            values.append(explainer.explain(budget=self.budget))
        return values


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
