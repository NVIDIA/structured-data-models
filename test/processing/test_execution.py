# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    EnsembleTable,
    Recipe,
    RelatedTables,
    Stype,
    TableTensor,
)
from sdm.processing.execution import RecipeExecution


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
            .fit(fitted_related[member_id])
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
            sp.Standardize().fit(fit_table).transform(query_related[member_id])
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
            .fit(fitted_related[member_id])
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
            .transform(query_related[member_id])
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


def test_member_context_exposes_input_stypes() -> None:
    x = TableTensor(
        columns={Stype.numerical: ("n",), Stype.categorical: ("c",)},
        numerical=torch.randn(3, 1),
        categorical=CategoricalTensor.from_tensor(torch.zeros(3, 1).long()),
    )
    y = torch.zeros(3, 1)
    recipe = Recipe(features=sp.ToNumerical())

    (context,) = RecipeExecution(recipe).fit_transform(
        x=x,
        y=y,
        related_tables=None,
    )

    assert context.input_stypes == x.stypes
    assert context.x.columns[Stype.numerical] == ("n", "c")
    assert context.x_context_only is None


@pytest.mark.parametrize("num_members", [1, 2])
@pytest.mark.parametrize("named", [False, True])
def test_context_only_features_use_independent_fit(
    num_members: int,
    named: bool,
) -> None:
    x = torch.tensor([[0.0], [2.0], [4.0]])
    x_context_only = torch.tensor([[10.0, 20.0], [14.0, 26.0], [18.0, 32.0]])
    query = torch.tensor([[6.0]])
    execution = RecipeExecution(Recipe(features=sp.Standardize()))
    columns = ["known", "past_a", "past_b"] if named else ["0", "1", "2"]
    inputs = TableTensor.from_tensor(
        torch.cat([x, x_context_only], dim=-1),
        columns=columns,
    )

    contexts = execution.fit_transform(
        x=inputs if named else inputs.numerical,
        y=torch.zeros(3, 1),
        related_tables=None,
        context_only_columns=columns[1:] if named else [1, 2],
        num_members=num_members,
    )
    query_table = TableTensor.from_tensor(query, columns=columns[:1])
    queries = execution.transform(x=query_table, related_tables=None)

    paired = inputs.select_columns(columns[:1])
    features = sp.Standardize().fit(paired)
    context_only = sp.Standardize().fit_transform(
        inputs.select_columns(columns[1:])
    )
    expected_query = features.transform(query_table)
    assert len(contexts) == len(queries) == num_members
    for context, transformed_query in zip(contexts, queries):
        assert context.x.equal(features.transform(paired))
        assert context.x_context_only is not None
        assert context.x_context_only.equal(context_only)
        assert transformed_query.x.equal(expected_query)


def test_context_only_columns_with_batched_inputs() -> None:
    x = torch.arange(18.0).reshape(2, 3, 3)
    y = torch.zeros(2, 3, 1)
    execution = RecipeExecution(Recipe(features=sp.Standardize()))
    contexts = execution.fit_transform(
        x=x,
        y=y,
        related_tables=None,
        context_only_columns=["1"],
    )
    query = TableTensor.from_tensor(x[:, :1, [0, 2]], columns=["0", "2"])
    queries = execution.transform(x=query, related_tables=None)

    assert len(contexts) == len(queries) == 2
    for i, (context, transformed_query) in enumerate(zip(contexts, queries)):
        table = TableTensor.from_tensor(x[i])
        paired = table.drop_columns(["1"])
        features = sp.Standardize().fit(paired)
        assert context.x.equal(features.transform(paired))
        assert context.y.equal(TableTensor.from_tensor(y[i]))
        assert context.x_context_only is not None
        assert context.x_context_only.equal(
            sp.Standardize().fit_transform(table.select_columns(["1"]))
        )
        assert transformed_query.x.equal(features.transform(query[i]))
