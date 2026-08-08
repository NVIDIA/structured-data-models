import pytest
import torch

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.processing import Processor, TableDispatch
from sdm.processing._recipe_execution import _RecipeExecution


class _Add(Processor):
    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.value)


def test_table_dispatch() -> None:
    x = TableTensor.from_tensor(torch.zeros(4, 1))

    execution = _RecipeExecution._bind(
        recipe=Recipe(features=TableDispatch(task=_Add(1), related=_Add(2))),
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

    (context,) = execution.contexts
    assert context.x.numerical.equal(x.numerical + 1)
    assert context.related_tables is not None
    assert context.related_tables.tables["x"].numerical.equal(x.numerical + 2)


def test_recipe_reject() -> None:
    with pytest.raises(ValueError, match="not supported"):
        Recipe(target=TableDispatch())
    with pytest.raises(ValueError, match="not supported"):
        Recipe(output=TableDispatch())
