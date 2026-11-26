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
        scoring: A function ``scoring(pred, target)`` that maps the predictions
            and the targets of the query rows to one number, for which
            higher is better, *e.g.*, an accuracy or the negative of an
            error. If ``None``, will use the negative log loss for
            categorical targets, and the negative mean squared error for
            numerical targets.
    """

    def __init__(
        self,
        scoring: Callable[[TableTensor, TableTensor], Tensor | float]
        | None = None,
    ) -> None:
        self.scoring = default_scoring if scoring is None else scoring

    def explain(
        self,
        model: ICLModel,
        x_context: TableTensor,
        y_context: TableTensor,
        x_query: TableTensor,
        y_query: TableTensor,
        **kwargs: Any,
    ) -> TableTensor:
        r"""Explain the importance of feature columns.

        Args:
            model: The model.
            x_context: The feature tensor of in-context examples with shape
                ``[..., R_context, D]`` with ``R_context`` rows and ``D``
                columns.
            y_context: The targets of in-context examples with shape
                ``[R_context, 1]``.
            x_query: The feature tensor of query examples with shape
                ``[..., R_query, D]`` with ``R_query`` rows and ``D`` columns.
            y_query: The targets of query examples with shape
                ``[R_query, 1]``.
            kwargs: Additional keyword arguments passed to the model call.

        Returns:
            The table of score decreases with shape ``[..., 1, D]``.
        """
        generator = kwargs.pop("generator", None)

        def score(*columns: str) -> Tensor | float:
            out = model(
                x_context=x_context.drop_columns(columns),
                y_context=y_context,
                x_query=x_query.drop_columns(columns),
                generator=None
                if generator is None
                else generator.clone_state(),
                **kwargs,
            )
            return self.scoring(out, y_query)

        ref = score()
        decreases = {
            name: torch.as_tensor(
                ref - score(name),
                device=x_query.device,
                dtype=x_query.dtype,
            )
            for columns in x_query.columns.values()
            for name in columns
        }

        return TableTensor.from_tensor(
            tensor=torch.stack(list(decreases.values()), dim=-1).unsqueeze(-2),
            columns=list(decreases),
        )
