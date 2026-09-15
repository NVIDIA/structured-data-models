# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import EnsembleTable, Recipe, RelatedTables, TableTensor
from sdm.models import KumoRelational
from sdm.processing.execution import RecipeExecution


def test_relational_recipe_uses_fitted_task_states_for_shared_query() -> None:
    values = torch.arange(8 * 32).reshape(8, 32).float() / 20
    x = TableTensor.from_tensor(
        torch.stack((values, values.square(), torch.ones_like(values)), dim=-1)
    )
    y = TableTensor.from_tensor(values.unsqueeze(-1))
    query_values = torch.arange(6).float() / 3
    query = torch.stack(
        (query_values, query_values.square(), torch.ones_like(query_values)),
        dim=-1,
    )
    execution = RecipeExecution(KumoRelational.default_recipe())
    execution.fit_transform(x=x, y=y, related_tables=None)
    expected = execution.transform(
        x=TableTensor.from_tensor(query.unsqueeze(0).expand(8, -1, -1)),
        related_tables=None,
    )

    shared_query = EnsembleTable.from_table(
        TableTensor.from_tensor(query), num_members=8
    )
    for _ in range(2):
        actual = execution.transform(x=shared_query, related_tables=None)
        assert len(actual) == 8
        for output, reference in zip(actual, expected, strict=True):
            assert output.x.size(-1) == 2
            assert output.x.equal(reference.x)


def test_sequence_uses_separate_task_states_for_shared_query() -> None:
    tables = (
        TableTensor.from_tensor(torch.tensor([[0.0], [2.0]])),
        TableTensor.from_tensor(torch.tensor([[10.0], [14.0], [18.0]])),
    )
    x = EnsembleTable.from_tables(tables, (0, 1))
    query = TableTensor.from_tensor(torch.tensor([[2.0], [6.0], [10.0]]))
    execution = RecipeExecution(Recipe(features=sp.Standardize()))
    execution.fit_transform(x=x, y=x, related_tables=None)
    actual = execution.transform(
        x=EnsembleTable.from_table(query, num_members=2), related_tables=None
    )

    for output, table in zip(actual, tables, strict=True):
        assert output.x.equal(sp.Standardize().fit(table).transform(query))


@pytest.mark.parametrize("num_members", [1, 3])
def test_sequence_rejects_changed_task_member_count(num_members: int) -> None:
    x = TableTensor.from_tensor(
        torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]])
    )
    execution = RecipeExecution(Recipe(features=sp.Standardize()))
    execution.fit_transform(x=x, y=x, related_tables=None)

    with pytest.raises(ValueError, match="same number of ensemble members"):
        execution.transform(
            x=EnsembleTable.from_table(x[0], num_members=num_members),
            related_tables=None,
        )


def test_sequence_uses_batched_fit_states_for_shared_query() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fitted_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]])
            ),
        ),
        locations=((0, 0), (0, 1)),
    )
    query_table = TableTensor.from_tensor(
        torch.tensor([[2.0], [6.0], [10.0], [14.0]])
    )
    query_related = EnsembleTable.from_table(query_table, num_members=2)
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )
    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    for member_id, query in enumerate(queries):
        assert query.related_tables is not None
        expected = (
            sp.Standardize()
            .fit(fitted_related.table(member_id))
            .transform(query_table)
        )
        assert query.related_tables.tables["x"].equal(expected)


def test_sequence_uses_separate_fit_states_for_shared_query() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fit_tables = (
        TableTensor.from_tensor(torch.tensor([[0.0], [2.0]])),
        TableTensor.from_tensor(torch.tensor([[10.0], [14.0], [18.0]])),
    )
    fitted_related = EnsembleTable.from_tables(fit_tables, (0, 1))
    query_table = TableTensor.from_tensor(
        torch.tensor([[2.0], [6.0], [10.0], [14.0]])
    )
    query_related = EnsembleTable.from_table(query_table, num_members=2)
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )
    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    for member_id, query in enumerate(queries):
        assert query.related_tables is not None
        expected = (
            sp.Standardize().fit(fit_tables[member_id]).transform(query_table)
        )
        assert query.related_tables.tables["x"].equal(expected)


def test_sequence_uses_shared_fit_state_for_batched_query() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fit_table = TableTensor.from_tensor(torch.tensor([[0.0], [2.0]]))
    fitted_related = EnsembleTable.from_table(fit_table, num_members=2)
    query_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor(
                    [
                        [[2.0], [6.0], [10.0]],
                        [[3.0], [6.0], [9.0]],
                    ]
                )
            ),
        ),
        locations=((0, 0), (0, 1)),
    )
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )
    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    for member_id, query in enumerate(queries):
        assert query.related_tables is not None
        expected = (
            sp.Standardize()
            .fit(fit_table)
            .transform(query_related.table(member_id))
        )
        assert query.related_tables.tables["x"].equal(expected)


def test_sequence_batches_separate_compatible_queries_for_fit_states() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fitted_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]])
            ),
        ),
        locations=((0, 0), (0, 1)),
    )
    query_tables = (
        TableTensor.from_tensor(torch.tensor([[2.0], [6.0], [10.0]])),
        TableTensor.from_tensor(torch.tensor([[3.0], [6.0], [9.0]])),
    )
    query_related = EnsembleTable.from_tables(query_tables, (0, 1))
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )
    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    for member_id, query in enumerate(queries):
        assert query.related_tables is not None
        expected = (
            sp.Standardize()
            .fit(fitted_related.table(member_id))
            .transform(query_tables[member_id])
        )
        assert query.related_tables.tables["x"].equal(expected)


def test_sequence_splits_batched_query_for_separate_fit_states() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fit_tables = (
        TableTensor.from_tensor(torch.tensor([[0.0], [2.0]])),
        TableTensor.from_tensor(torch.tensor([[10.0], [14.0], [18.0]])),
    )
    fitted_related = EnsembleTable.from_tables(fit_tables, (0, 1))
    query_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor(
                    [
                        [[2.0], [6.0], [10.0]],
                        [[3.0], [6.0], [9.0]],
                    ]
                )
            ),
        ),
        locations=((0, 0), (0, 1)),
    )
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )
    queries = execution.transform(
        x=x,
        related_tables=RelatedTables(
            tables={"x": query_related},
            relationships=[],
            task_links=[],
        ),
    )

    for member_id, query in enumerate(queries):
        assert query.related_tables is not None
        expected = (
            sp.Standardize()
            .fit(fit_tables[member_id])
            .transform(query_related.table(member_id))
        )
        assert query.related_tables.tables["x"].equal(expected)


def test_sequence_rejects_split_state_in_multi_state_fit_group() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fitted_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]])
            ),
        ),
        locations=((0, 0), (0, 0), (0, 1)),
    )
    query_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor(
                    [
                        [[2.0], [6.0], [10.0]],
                        [[3.0], [6.0], [9.0]],
                        [[4.0], [8.0], [12.0]],
                    ]
                )
            ),
        ),
        locations=((0, 0), (0, 1), (0, 2)),
    )
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=3,
    )

    with pytest.raises(ValueError, match="Cannot align query tables"):
        execution.transform(
            x=x,
            related_tables=RelatedTables(
                tables={"x": query_related},
                relationships=[],
                task_links=[],
            ),
        )


def test_sequence_rejects_queries_that_cannot_form_fitted_batch() -> None:
    x = TableTensor.from_tensor(torch.zeros(2, 1))
    fitted_related = EnsembleTable(
        groups=(
            TableTensor.from_tensor(
                torch.tensor([[[0.0], [2.0]], [[10.0], [14.0]]])
            ),
        ),
        locations=((0, 0), (0, 1)),
    )
    query_related = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(torch.tensor([[2.0], [6.0], [10.0]])),
            TableTensor.from_tensor(
                torch.tensor([[3.0], [6.0], [9.0], [12.0]])
            ),
        ),
        member_table_ids=(0, 1),
    )
    execution = RecipeExecution(
        Recipe(
            features=sp.Sequential(sp.TableDispatch(related=sp.Standardize()))
        )
    )

    execution.fit_transform(
        x=x,
        y=x,
        related_tables=RelatedTables(
            tables={"x": fitted_related},
            relationships=[],
            task_links=[],
        ),
        num_members=2,
    )

    with pytest.raises(ValueError, match="Cannot align query tables"):
        execution.transform(
            x=x,
            related_tables=RelatedTables(
                tables={"x": query_related},
                relationships=[],
                task_links=[],
            ),
        )
