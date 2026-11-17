# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor
from sdm.prior import Correlated, CorrelatedContext, LogUniform, UniformInt
from sdm.prior.functions import RandomFunction
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._scale import _standardize
from sdm.prior.tabular.transform._table import (
    categorical_mask,
    feature_values,
)
from sdm.prior.util import random_weight


class Relabel(PriorTransform):
    """Recompute the target from selected feature columns.

    Equal inputs receive equal scores. Classification retains the input
    target when the scores cannot separate the declared classes.

    Args:
        random_function: Function used to score the selected features.
        exact_probability: Probability that classification uses exact score
            cuts rather than a noisy score rank.
    """

    def __init__(
        self,
        random_function: RandomFunction,
        *,
        exact_probability: float = 0.5,
    ) -> None:
        self.random_function = random_function
        self.exact_probability = exact_probability
        self.bias_floor = Correlated(
            LogUniform(0.5, 6.0), key="Relabel.bias_floor"
        )
        self.temperature = Correlated(
            LogUniform(20.0, 200.0), key="Relabel.temperature"
        )

    def _apply(
        self,
        x: Tensor,
        y: Tensor,
        cat_mask: Tensor,
        num_classes: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        num_columns = x.size(-1)
        with CorrelatedContext(generator=generator):
            k = UniformInt(1, min(4, num_columns)).sample(generator=generator)
            # Categorical columns rank after every numerical one:
            order = torch.rand(
                num_columns, device=x.device, generator=generator
            ) + cat_mask.to(x.dtype)
            inputs = x[:, order.argsort()[:k]]
            score = self.random_function.forward(
                x=inputs, out_channels=1, generator=generator
            )
            score = _standardize(score)[:, 0]
            # Give equal input rows the same score despite rounding.
            position = torch.arange(inputs.size(0), device=inputs.device)
            order = position
            for j in reversed(range(k)):  # Lexicographic row order.
                key = inputs[..., j].gather(-1, order)
                order = order.gather(-1, key.sort(dim=-1, stable=True).indices)
            rows = inputs[order]
            starts = torch.ones_like(order, dtype=torch.bool)
            starts[1:] = (rows[1:] != rows[:-1]).any(dim=-1)
            first = torch.cummax(
                torch.where(starts, position, 0), dim=-1
            ).values
            tied = score.gather(-1, order).gather(-1, first)
            score = torch.empty_like(score).scatter_(-1, order, tied)
            if y.is_floating_point():
                return score.to(y.dtype)
            labels, separable = self._classify(
                score=score, num_classes=num_classes, generator=generator
            )
        return labels.to(y.dtype) if separable else y

    def _classify(
        self,
        score: Tensor,  # [num_rows]
        num_classes: int,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool]:
        num_rows = score.size(0)
        ranked = score.sort(dim=-1).values
        position = torch.arange(num_rows, device=score.device)
        starts = torch.ones_like(ranked, dtype=torch.bool)
        starts[1:] = ranked.diff(dim=-1) > 0
        start = torch.where(starts, position, num_rows).sort(dim=-1).values
        distinct = starts.sum(dim=-1)
        floor = self.bias_floor.sample_batch(generator=generator).to(
            score.dtype
        )
        bias = random_weight(
            size=(num_classes,), dtype=score.dtype, generator=generator
        )
        bias = torch.maximum((bias + 1e-4).log(), -floor)
        share = torch.softmax(bias, dim=-1).cumsum(dim=-1)[:-1]

        # Cut after the run of equal scores holding each share quantile; the
        # cuts are pushed apart onto distinct run boundaries so every class
        # keeps a run whenever the score has at least ``num_classes`` values:
        offset = torch.arange(num_classes - 1, device=score.device)
        cut = torch.searchsorted(
            sorted_sequence=start,
            input=(share * (num_rows - 1)).long(),
            right=True,
        )
        cut = torch.cummax(cut - offset, dim=-1).values + offset
        cut = torch.minimum(
            input=cut,
            other=distinct - num_classes + 1 + offset,
        )
        row = start.gather(-1, cut.clamp(0, num_rows - 1))
        threshold = 0.5 * (
            ranked.gather(-1, (row - 1).clamp(min=0))
            + ranked.gather(-1, row.clamp(max=num_rows - 1))
        )

        # Jitter in tie-averaged rank space, then read the score back at the
        # jittered rank: flips concentrate at the cuts regardless of the
        # score's scale, tail, or ties.
        a = self.temperature.sample_batch(generator=generator).to(score.dtype)
        rank = 0.5 * (
            torch.searchsorted(ranked, score)
            + torch.searchsorted(ranked, score, right=True)
            - 1
        )
        jitter = torch.randn(
            score.shape,
            dtype=score.dtype,
            device=score.device,
            generator=generator,
        ) * (0.5 * num_rows / (a * (num_classes - 1)))
        index = (rank + jitter).round().long().clamp(0, num_rows - 1)
        is_exact = (
            torch.rand((), device=score.device, generator=generator)
            < self.exact_probability
        )
        score = torch.where(
            condition=is_exact, input=score, other=ranked.gather(-1, index)
        )
        labels = (score.unsqueeze(-1) > threshold).sum(dim=-1)
        return labels, bool(distinct >= num_classes)

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = feature_values(x)
        cat_mask = categorical_mask(x)
        if y.numerical.size(-1):
            target = self._apply(
                x=values,
                y=y.numerical[:, 0],
                cat_mask=cat_mask,
                num_classes=1,
                generator=generator,
            )
            return x, y.replace_blocks(numerical=target[:, None])
        categories = y.categorical.categories
        code = self._apply(
            x=values,
            y=y.categorical.code[:, 0],
            cat_mask=cat_mask,
            num_classes=len(categories[0]),
            generator=generator,
        )
        return x, y.replace_blocks(
            categorical=CategoricalTensor(
                code=code[:, None], categories=categories
            ),
        )
