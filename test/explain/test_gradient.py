# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.explain import GradientExplainer
from sdm.models import ICLModel


class _LinearModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_related_tables = True

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        table = x_context if x_query is None else x_query
        assert table is not None
        numerical = 2.0 * table.numerical
        if related_query_tables is not None:
            numerical = (
                numerical
                + 3.0 * related_query_tables.tables["related"].numerical
            )
        return table.replace_blocks(numerical=numerical)

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()


@pytest.mark.parametrize("fitted", [False, True])
def test_returns_query_input_gradients(fitted: bool) -> None:
    model = _LinearModel()
    x_context = torch.zeros(1, 2)
    y_context = torch.zeros(1, 1)
    x_query = torch.ones(1, 2)
    related_tables = RelatedTables(
        tables={
            "related": TableTensor(numerical=torch.ones(1, 2)),
            "unused": TableTensor(numerical=torch.ones(1, 2)),
        },
        relationships=(
            {
                "left_table": "related",
                "left_column": "0",
                "right_table": "unused",
                "right_column": "0",
            },
        ),
        task_links=(
            {"task_column": "0", "table": "related", "table_column": "0"},
        ),
    )
    explainer = GradientExplainer(
        output=lambda prediction: prediction.numerical
    )

    if fitted:
        model.fit(x_context, y_context, related_tables)
        result = explainer.explain(model, x_query, related_tables)
    else:
        result = explainer.explain(
            model,
            x_query,
            related_tables,
            x_context=x_context,
            y_context=y_context,
            related_context_tables=related_tables,
        )

    torch.testing.assert_close(
        result.x.numerical, torch.full_like(x_query, 2.0)
    )
    assert result.related_tables is not None
    torch.testing.assert_close(
        result.related_tables.tables["related"].numerical,
        torch.full_like(x_query, 3.0),
    )
    torch.testing.assert_close(
        result.related_tables.tables["unused"].numerical,
        torch.zeros_like(x_query),
    )
    assert result.related_tables.relationships == related_tables.relationships
    assert result.related_tables.task_links == related_tables.task_links
