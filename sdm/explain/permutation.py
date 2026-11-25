# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Callable

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.explain._tabular import _score, _to_table
from sdm.models import ICLModel
from sdm.processing import ToNumerical


class PermutationExplainer:
    r"""Compute the score decrease from shuffling feature columns from the
    `"Random Forests" <https://doi.org/10.1023/A:1010933404324>`_ paper.

    Each column is in turn shuffled across the query rows of a fitted model,
    which breaks its association with the target.

    Args:
        num_repeats: The number of shuffles ``N`` per column.
        score: A function ``score(pred, target)`` that maps the predictions
            and the targets of the query rows to one number, for which
            higher is better, *e.g.*, an accuracy or the negative of an
            error. If ``None``, the negative log loss for a categorical
            target, and the negative squared error of the mean over all
            predicted columns otherwise.
    """

    def __init__(
        self,
        num_repeats: int = 5,
        score: Callable[[TableTensor, TableTensor], Tensor | float]
        | None = None,
    ) -> None:
        self.num_repeats = num_repeats
        self.score = _score if score is None else score

    def explain(
        self,
        model: ICLModel,
        x_query: TableTensor,
        *,
        y_query: TableTensor,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        r"""Explain the score of query examples.

        Args:
            model: The fitted model.
            x_query: The feature table of query examples with shape
                ``[R, D]`` with ``R`` rows and ``D`` columns.
            y_query: The targets of query examples with shape ``[R, 1]``.
            generator: Pseudorandom number generator used for shuffling.

        Returns:
            The table of score decreases with shape ``[N, D]``, named by the
            numerical and then categorical columns.
        """
        inputs = ToNumerical().transform(x_query).numerical  # Codes.
        baseline = self.score(model.predict(x_query), y_query)
        decreases: list[list[Tensor | float]] = []
        for j in range(inputs.size(-1)):
            perm = torch.rand(
                size=(self.num_repeats, len(x_query)),
                generator=generator,
                device=x_query.device,
            ).argsort(dim=-1)  # [N, R]
            values = inputs.repeat(self.num_repeats, 1)  # [N * R, D]
            values[:, j] = inputs[:, j][perm].flatten()
            pred = model.predict(_to_table(values, like=x_query))
            decreases.append(
                [  # Score the slice `[R, O]` of each repeat:
                    baseline - self.score(pred[i : i + len(x_query)], y_query)
                    for i in range(0, len(pred), len(x_query))
                ]
            )
        return TableTensor.from_tensor(
            tensor=inputs.new_tensor(decreases).T,  # [D, N] -> [N, D]
            columns=(*x_query.numerical_columns, *x_query.categorical_columns),
        )
