# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain.permutation import default_scoring
from sdm.models import ICLModel


class LOCOExplainer:
    r"""Compute the leave-one-covariate-out (LOCO) importance of feature
    columns, also known as leave-one-feature-out (LOFO) importance, from the
    `"Distribution-Free Predictive Inference For Regression"
    <https://arxiv.org/abs/1604.04173>`_ paper.

    Each column is in turn dropped from both the in-context and the query
    examples, so that the model predicts as if it had never seen it. The
    model is called once with all columns and once without each column, and
    the importance of a column is the score decrease without it.

    Args:
        score: A function ``score(pred, target)`` that maps the predictions
            and the targets of the query rows to one number, for which
            higher is better, *e.g.*, an accuracy or the negative of an
            error. If ``None``, the negative log loss for a categorical
            target, and the negative squared error of the mean over all
            predicted columns otherwise.
    """

    def __init__(
        self,
        score: Callable[[TableTensor, TableTensor], Tensor | float]
        | None = None,
    ) -> None:
        self.score = default_scoring if score is None else score

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
        *,
        y_query: TableTensor,
        x_context: TableTensor,
        y_context: TableTensor,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> TableTensor:
        r"""Explain the score of query examples.

        Args:
            model: The model. Its fitted state is neither used nor changed,
                so repeat the arguments of its fit to explain the same
                predictions.
            x_query: The feature table of query examples with shape
                ``[R_query, D]`` with ``R_query`` rows and ``D`` columns.
            y_query: The targets of query examples with shape
                ``[R_query, 1]``.
            x_context: The feature table of in-context examples with shape
                ``[R_context, D]``.
            y_context: The targets of in-context examples with shape
                ``[R_context, 1]``.
            generator: Pseudorandom number generator used for the model calls,
                which all start from its current state and leave it unchanged.
                Without it, each model call samples its ensemble anew.
            kwargs: Additional keyword arguments passed to the model call,
                *e.g.*, ``num_estimators``.

        Returns:
            The table of score decreases with shape ``[1, D]``, named by the
            numerical and then categorical columns.
        """
        names = (*x_query.numerical_columns, *x_query.categorical_columns)

        def score_without(*columns: str) -> Tensor | float:
            state = None if generator is None else generator.clone_state()
            pred = model(
                x_context=x_context.drop_columns(columns),
                y_context=y_context,
                x_query=x_query.drop_columns(columns),
                generator=state,
                **kwargs,
            )
            return self.score(pred, y_query)

        baseline = score_without()
        decreases = [baseline - score_without(name) for name in names]
        return TableTensor.from_tensor(
            tensor=x_query.numerical.new_tensor([decreases]),  # [1, D]
            columns=names,
        )
