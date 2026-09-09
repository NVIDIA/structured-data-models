from typing import cast

import pytest
import torch

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.processing import Processor, Standardize, TableDispatch
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable


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


def test_recipe_reject() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Recipe(target=TableDispatch())
    with pytest.raises(ValueError, match="not supported"):
        Recipe(output=TableDispatch())


def test_recipe_uses_fitted_related_table_groups_during_transform() -> None:
    context_tables = (
        TableTensor.from_tensor(
            torch.tensor([[1.0], [3.0]]), columns=("first",)
        ),
        TableTensor.from_tensor(
            torch.tensor([[10.0], [20.0], [30.0]]), columns=("second",)
        ),
    )
    context = EnsembleTable.from_tables(
        tables=context_tables,
        member_table_ids=(0, 1),
    )
    query_tables = (
        TableTensor.from_tensor(torch.tensor([[4.0], [6.0]])),
        TableTensor.from_tensor(torch.tensor([[40.0], [60.0]])),
    )
    query = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(query_tables)),),
        locations=((0, 0), (0, 1)),
    )
    task = TableTensor.from_tensor(torch.zeros(2, 1))
    execution = RecipeExecution(
        Recipe(
            features=TableDispatch(
                related=Standardize(with_std=False),
            )
        )
    )
    execution.fit_transform(
        x=task,
        y=task,
        related_tables=RelatedTables(
            tables={"related": context},
            relationships=(),
            task_links=(),
        ),
        num_members=2,
    )

    outputs = execution.transform(
        x=task,
        related_tables=RelatedTables(
            tables={"related": query},
            relationships=(),
            task_links=(),
        ),
    )

    assert outputs[0].related_tables is not None
    assert outputs[1].related_tables is not None
    assert outputs[0].related_tables.tables["related"].numerical.tolist() == [
        [2.0],
        [4.0],
    ]
    assert outputs[1].related_tables.tables["related"].numerical.tolist() == [
        [20.0],
        [40.0],
    ]
