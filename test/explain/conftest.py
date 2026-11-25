# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from torch import Tensor

import sdm.processing as sp
from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel


class ScoreModel(ICLModel):
    """Predict a function of the query columns, ignoring the context rows.

    The function receives the present columns by name, with categorical
    columns as their codes. Numerical targets predict the function value as
    column ``"pred"``, between the columns ``"low"`` and ``"high"`` one below
    and above it. Categorical targets predict it as the probability of the
    second of two classes, whose column comes first. With ``noise``, the
    generator of a model call shifts all of its predictions by one random
    draw.
    """

    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_multi_target = False
    supports_context_only_columns = False
    supports_related_tables = False

    def __init__(
        self,
        fn: Callable[[dict[str, Tensor]], Tensor | float],
        *,
        noise: bool = False,
    ) -> None:
        super().__init__(task=None)
        self.fn = fn
        self.noise = noise
        self.eval()

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if y_context is None:
            assert cache is not None
            classes = cast("Tensor | None", cache["classes"])
        elif y_context.categorical.size(-1) > 0:
            classes = y_context.categorical.categories[0]
        else:
            classes = None

        table = x_context if x_query is None else x_query
        assert table is not None
        columns = dict(
            zip(table.numerical_columns, table.numerical.unbind(-1))
        )
        score = table.numerical.new_zeros(table.size()[:-1])  # [R]
        score = score + self.fn(columns)
        if self.noise:
            score = score + torch.rand(
                size=(),
                generator=generator,
                device=score.device,
            )
        if classes is None:
            return TableTensor.from_tensor(
                tensor=torch.stack([score - 1.0, score, score + 1.0], dim=-1),
                columns=["low", "pred", "high"],
            )
        return TableTensor.from_tensor(
            tensor=torch.stack([score, 1.0 - score], dim=-1),
            columns=[str(value) for value in reversed(classes.tolist())],
        )

    @classmethod
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe(
            features=sp.ToNumerical(),
            output=sp.AverageEstimators(),
        )


@pytest.fixture
def score_model() -> type[ScoreModel]:
    return ScoreModel
