# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.processing.execution import RecipeExecution

if TYPE_CHECKING:
    from shapiq import InteractionValues


class ShapleyExplainer:
    r"""Compute Shapley values of feature columns from the `"A Unified
    Approach to Interpreting Model Predictions"
    <https://arxiv.org/abs/1705.07874>`_ paper.

    A fitted model is evaluated on subsets of the columns of each query row,
    and the columns outside a subset are removed by imputation.
    The resulting attributions describe each feature's contribution to the
    difference between the row's prediction and the baseline prediction.

    .. note::
        :class:`ShapleyExplainer` requires ``shapiq>=1.8``.
        See `here <https://shapiq.readthedocs.io>`_ on how to install.

    Args:
        kwargs: Additional keyword arguments passed to
            ``shapiq.TabularExplainer``, *e.g.*, ``imputer``, ``index``
            or ``max_order``.
    """

    def __init__(self, **kwargs: Any) -> None:
        if "model" in kwargs:
            raise ValueError(
                "'model' should be supplied by 'explain(model=...)'"
            )
        if "data" in kwargs:
            raise ValueError(
                "'data' should be supplied by 'explain(background=...)'"
            )
        if "categorical_features" in kwargs:
            raise ValueError(
                "'categorical_features' are supplied automatically"
            )
        if "random_state" in kwargs:
            raise ValueError(
                "'random_state' should be supplied by 'explain(generator=...)'"
            )
        self.kwargs = kwargs

    def explain(
        self,
        model: ICLModel,
        x: TableTensor,
        output: str,
        background: TableTensor,
        budget: int,
        *,
        generator: torch.Generator | None = None,
    ) -> list[InteractionValues]:
        r"""Explain the contributions of preprocessed features.

        Args:
            model: The fitted model.
            x: The feature tensor of query examples with shape ``[R, D]``
                with ``R`` rows and ``D`` columns.
            output: The name of the predicted column to explain.
            background: The feature tensor of reference rows with shape
                ``[R_background, D]``, *e.g.*, the in-context examples.
            budget: The budget to use for the approximation.
            generator: Pseudorandom number generator used for sampling column
                subsets and reference rows.

        Returns:
            The ``shapiq.InteractionValues`` of each query row.
        """
        from shapiq import TabularExplainer

        if model._cache is None:
            raise RuntimeError(
                f"{model.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{model.__class__.__name__}.fit()' before."
            )
        recipe_execution = cast(
            RecipeExecution,
            model._cache["recipe_execution"],
        )
        if recipe_execution.num_members != 1:
            raise RuntimeError(
                f"{self.__class__.__name__!r} requires exactly one estimator "
                f"in {model.__class__.__name__!r}"
            )

        x = recipe_execution.transform(x, related_tables=None)[0].x
        background = recipe_execution.transform(background, None)[0].x

        data = background.numerical.cpu().numpy()
        if len(data) == 1:
            data = data.repeat(2, axis=0)

        mask = cast(Tensor, cast(Cache, model._cache[0])["categorical_mask"])
        random_state: int | None = None
        if generator is not None:
            random_state = int(
                torch.randint(2**31, (), generator=generator, device=x.device)
            )

        def predict(values: np.ndarray) -> np.ndarray:
            values = values.astype(data.dtype, copy=False)
            table = TableTensor(
                columns={Stype.numerical: x.columns[Stype.numerical]},
                numerical=torch.from_numpy(values).to(x.device),
            )
            processor = recipe_execution.recipe.features
            try:
                recipe_execution.recipe.features = None
                out = model.predict(table)
            finally:
                recipe_execution.recipe.features = processor
            return out[output].numerical.squeeze(-1).cpu().numpy()

        explainer = TabularExplainer(
            model=predict,
            data=data,
            categorical_features=mask.nonzero().view(-1).tolist(),
            random_state=random_state,
            **self.kwargs,
        )
        values = x.numerical.cpu().numpy()
        return list(explainer.explain_X(values, budget=budget))


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
