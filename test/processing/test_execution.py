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
from sdm.testing import withCUDA


@withCUDA
def test_output_encoder_uses_raw_targets(device: torch.device) -> None:
    class AddTargetMean(sp.TargetAwareEnsembleProcessor):
        handles_stypes = (Stype.numerical,)
        requires_fit = False

        def _transform_ensemble(
            self,
            ensemble_table: EnsembleTable,
        ) -> EnsembleTable:
            assert self._target is not None
            mean = self._target.numerical.mean(dim=-2, keepdim=True)
            return ensemble_table.replace_groups(
                [
                    group.replace_blocks(numerical=group.numerical + mean)
                    for group in ensemble_table._iter_groups()
                ]
            )

    execution = RecipeExecution(
        Recipe(
            target=sp.FlipSign(probability=1.0),
            output=[sp.AverageEstimators(), AddTargetMean()],
        )
    )
    x = torch.zeros(2, 3, 0, device=device)
    y = torch.tensor(
        [[[1.0], [3.0], [5.0]], [[10.0], [30.0], [50.0]]], device=device
    )
    outputs = [
        TableTensor.from_tensor(torch.full((2, 1, 1), value, device=device))
        for value in (1.0, 3.0)
    ]
    for target in (y, -y):
        execution.fit_transform(
            x=x,
            y=target,
            related_tables=None,
            num_members=2,
        )
        result = execution.transform_output(outputs)
        torch.testing.assert_close(
            result.numerical,
            target.mean(dim=-2, keepdim=True) + 2.0,
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
    assert context.x.numerical_columns == ("n", "c")
