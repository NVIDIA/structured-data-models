# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

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
        r"""Explain the contributions of feature columns to predictions.

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
