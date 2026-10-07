# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.explain import GradientExplainer
from sdm.models import ICLModel


class _LinearModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_multi_target = False
    supports_related_tables = True

    def __init__(self) -> None:
        super().__init__(task=None)
        self.eval()

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
    explainer = GradientExplainer()

    if fitted:
        model.fit(x_context, y_context, related_tables)
        x_attributions, related_attributions = explainer.explain(
            model, x_query, related_tables
        )
    else:
        x_attributions, related_attributions = explainer.explain(
            model,
            x_query,
            related_tables,
            x_context=x_context,
            y_context=y_context,
            related_context_tables=related_tables,
        )

    torch.testing.assert_close(
        x_attributions.numerical, (2.0 * torch.eye(2)).unsqueeze(1)
    )
    assert related_attributions is not None
    torch.testing.assert_close(
        related_attributions.tables["related"].numerical,
        (3.0 * torch.eye(2)).unsqueeze(1),
    )
    torch.testing.assert_close(
        related_attributions.tables["unused"].numerical,
        torch.zeros(2, 1, 2),
    )
    assert related_attributions.relationships == related_tables.relationships
    assert related_attributions.task_links == related_tables.task_links
    if not fitted:
        assert model._cache is None


@pytest.mark.parametrize("fitted", [False, True])
def test_differentiates_final_prediction(fitted: bool) -> None:
    model = _LinearModel()
    x_context = torch.zeros(1, 2)
    y_context = torch.zeros(1, 1)
    x_query = torch.tensor([[0.0, 1.0]])
    recipe = Recipe(output=sp.Softmax())
    explainer = GradientExplainer()
    if fitted:
        model.fit(x_context, y_context, recipe=recipe)
        x_attributions, related_attributions = explainer.explain(
            model, x_query
        )
    else:
        x_attributions, related_attributions = explainer.explain(
            model,
            x_query,
            x_context=x_context,
            y_context=y_context,
            recipe=recipe,
        )

    probabilities = (2.0 * x_query[0]).softmax(dim=-1)
    expected = 2.0 * (
        probabilities.diag() - probabilities.outer(probabilities)
    )
    torch.testing.assert_close(x_attributions.numerical, expected.unsqueeze(1))
    assert related_attributions is None


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_target_inversion_scales_preprocessed_input_gradients(
    fitted: bool,
    invert: bool,
) -> None:
    model = _LinearModel()
    x_context = torch.tensor([[-2.0, -4.0], [2.0, 4.0]])
    y_context = torch.tensor([[8.0], [12.0]])
    x_query = torch.tensor([[2.0, 4.0]])
    recipe = Recipe(
        features=sp.Standardize(),
        target=sp.Standardize(),
        output=[sp.InvertTarget()] if invert else [],
    )
    explainer = GradientExplainer()
    if fitted:
        model.fit(x_context, y_context, recipe=recipe)
        attributions, _ = explainer.explain(model, x_query)
    else:
        attributions, _ = explainer.explain(
            model,
            x_query,
            x_context=x_context,
            y_context=y_context,
            recipe=recipe,
        )

    expected = (4.0 if invert else 2.0) * torch.eye(2)
    torch.testing.assert_close(attributions.numerical, expected.unsqueeze(1))
