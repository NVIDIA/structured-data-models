# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import EnsembleTable, Recipe, RelatedTables, Stype, TableTensor
from sdm.processing import Processor, Standardize, TableDispatch
from sdm.processing.execution import RecipeExecution


class _Add(Processor):
    requires_fit = False
    handles_stypes = frozenset({Stype.numerical})

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.value)


def test_table_dispatch() -> None:
    x = TableTensor.from_tensor(torch.zeros(4, 1))

    recipe = Recipe(features=TableDispatch(task=_Add(1), related=_Add(2)))
    execution = RecipeExecution(recipe)
    (context,) = execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": x},
            relationships=[],
            task_links=[],
        ),
        num_members=1,
        generator=None,
    )
    assert context.x.numerical.equal(x.numerical + 1)
    assert context.related_tables is not None
    assert context.related_tables.tables["x"].numerical.equal(x.numerical + 2)


def test_related_query_uses_fitted_groups_when_row_count_changes() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fit_tables = (
        TableTensor.from_tensor(torch.tensor([[0.0], [2.0]])),
        TableTensor.from_tensor(torch.tensor([[10.0], [14.0]])),
        TableTensor.from_tensor(torch.tensor([[20.0], [25.0], [30.0]])),
    )
    fitted_related = EnsembleTable(
        groups=(
            cast(TableTensor, torch.stack(fit_tables[:2])),
            cast(TableTensor, fit_tables[2].unsqueeze(0)),
        ),
        locations=((0, 0), (0, 1), (1, 0), (1, 0)),
    )
    query_tables = (
        TableTensor.from_tensor(torch.tensor([[2.0], [6.0], [10.0], [14.0]])),
        TableTensor.from_tensor(torch.tensor([[3.0], [6.0], [9.0], [12.0]])),
        TableTensor.from_tensor(torch.tensor([[4.0], [8.0], [12.0], [16.0]])),
    )
    query_related = EnsembleTable(
        groups=(
            cast(TableTensor, query_tables[0].unsqueeze(0)),
            cast(TableTensor, torch.stack(query_tables[1:])),
        ),
        locations=((0, 0), (0, 0), (1, 0), (1, 1)),
    )
    execution = RecipeExecution(
        Recipe(features=TableDispatch(related=Standardize()))
    )
    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=4,
    )

    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    fit_table_by_member = (*fit_tables, fit_tables[2])
    query_table_by_member = (
        query_tables[0],
        query_tables[0],
        query_tables[1],
        query_tables[2],
    )
    for query, fit_table, query_table in zip(
        queries,
        fit_table_by_member,
        query_table_by_member,
        strict=True,
    ):
        assert query.related_tables is not None
        expected = Standardize().fit(fit_table).transform(query_table)
        assert query.related_tables.tables["x"].equal(expected)


def test_recipe_reject() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Recipe(target=TableDispatch())
    with pytest.raises(ValueError, match="not supported"):
        Recipe(output=TableDispatch())
