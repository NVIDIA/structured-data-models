# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.explain import GradientExplainer
from sdm.models import ICLModel
from sdm.models.callback import CaptureInputs, EnableInputGradients


class _LinearModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
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


@pytest.mark.parametrize("enable_gradients", [False, True])
def test_capture_inputs(enable_gradients: bool) -> None:
    model = _LinearModel()
    capture_inputs = CaptureInputs()
    callbacks = (
        (EnableInputGradients(), capture_inputs)
        if enable_gradients
        else (capture_inputs,)
    )
    related_tables = RelatedTables(
        tables={"related": TableTensor(numerical=torch.ones(1, 2))},
        relationships=(),
        task_links=(),
    )

    model(
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.ones(1, 2),
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        callbacks=callbacks,
    )

    assert len(capture_inputs.inputs) == 1
    x, related = capture_inputs.inputs[0]
    assert x.numerical.requires_grad is enable_gradients
    assert related is not None
    assert (
        related.tables["related"].numerical.requires_grad is enable_gradients
    )


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
    explainer = GradientExplainer()

    if fitted:
        model.fit(x_context, y_context, related_tables)
        x, related = explainer.explain(model, x_query, related_tables)
        assert model._cache is not None
    else:
        x, related = explainer.explain(
            model,
            x_query,
            related_tables,
            x_context=x_context,
            y_context=y_context,
            related_context_tables=related_tables,
            callbacks=(),
        )
        assert model._cache is None

    torch.testing.assert_close(
        x.numerical,
        2.0 * torch.eye(2).unsqueeze(1),
    )
    assert related is not None
    torch.testing.assert_close(
        related.tables["related"].numerical,
        3.0 * torch.eye(2).unsqueeze(1),
    )
    torch.testing.assert_close(
        related.tables["unused"].numerical,
        torch.zeros(2, 1, 2),
    )
    assert related.relationships == related_tables.relationships
    assert related.task_links == related_tables.task_links
