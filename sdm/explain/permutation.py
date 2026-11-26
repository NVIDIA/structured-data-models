# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.evaluation import to_class_indices
from sdm.models import ICLModel


def default_scoring(  # noqa: D103
    pred: TableTensor,
    target: TableTensor,
) -> Tensor:
    if target.categorical.size(-1) > 0:  # Log loss:
        prob, index = to_class_indices(pred, target, missing_score=0.0)
        prob = prob.gather(-1, index.unsqueeze(-1)).squeeze(-1)  # [R]
        return prob.clamp(min=torch.finfo(prob.dtype).tiny).log().mean(dim=-1)
    mean = pred.numerical.mean(dim=-1)  # [R]
    return -(mean - target.numerical[..., 0]).square().mean(dim=-1)


class PermutationExplainer:
    r"""Compute the score decrease from shuffling feature columns from the
    `"Random Forests" <https://doi.org/10.1023/A:1010933404324>`_ paper.

    Each column is in turn shuffled across the query rows of a fitted model.

    Args:
        num_repeats: The number of shuffles ``N`` per column.
        scoring: A function ``scoring(pred, target)`` that maps the predictions
            and the targets of the query rows to one number, for which
            higher is better, *e.g.*, an accuracy or the negative of an
            error. If ``None``, will use the negative log loss for
            categorical targets, and the negative mean squared error for
            numerical targets.
    """

    def __init__(
        self,
        num_repeats: int = 5,
        scoring: Callable[[TableTensor, TableTensor], Tensor | float]
        | None = None,
    ) -> None:
        self.num_repeats = num_repeats
        self.scoring = default_scoring if scoring is None else scoring

    def explain(
        self,
        model: ICLModel,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        r"""Explain the importance of feature columns.

        Args:
            model: The fitted model.
            x: The feature tensor of query examples with shape ``[..., R, D]``
                with ``R`` rows and ``D`` columns.
            y: The targets of query examples with shape ``[R, 1]``.
            generator: Pseudorandom number generator used for shuffling.

        Returns:
            The table of score decreases with shape ``[..., N, D]``.
        """
        ref = self.scoring(model.predict(x), y)

        perm = [
            torch.randperm(x.size(-2), device=x.device, generator=generator)
            for _ in range(self.num_repeats)
        ]
        perm = torch.cat(perm, dim=0)
        x = cast(TableTensor, torch.cat([x] * self.num_repeats, dim=-2))

        decreases: dict[str, Tensor] = {}
        for stype, block in x.items():
            for i, name in enumerate(x.columns[stype]):
                before, column, after = block.tensor_split([i, i + 1], dim=-1)
                column = column[..., perm, :]
                x_i = x.replace_blocks(
                    **{stype: torch.cat([before, column, after], dim=-1)},
                )
                out = model.predict(x_i)
                out = out.unflatten(-2, (self.num_repeats, -1))
                decreases[name] = torch.stack(
                    [
                        torch.as_tensor(
                            ref - self.scoring(out[..., i, :, :], y),
                            device=out.device,
                            dtype=out.numerical.dtype,
                        )
                        for i in range(self.num_repeats)
                    ],
                    dim=-1,
                )

        return TableTensor.from_tensor(
            tensor=torch.stack(list(decreases.values()), dim=-1),
            columns=list(decreases),
        )
