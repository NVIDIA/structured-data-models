import pytest
import torch

import sdm.processing as sp
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.processing import Processor
from sdm.processing._recipe_execution import _RecipeExecution


class _Add(Processor):
    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.value)


def _table(
    values: list[float],
    ids: list[int],
    *,
    value_column: str,
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: (value_column,),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor(values).unsqueeze(-1),
        id=ColumnarTensor((torch.tensor(ids),)),
    )


def _related_tables(*, query: bool) -> RelatedTables:
    if query:
        users = _table([30.0], [3], value_column="age")
        orders = _table([106.0], [3], value_column="amount")
    else:
        users = _table([10.0, 20.0], [1, 2], value_column="age")
        orders = _table([100.0, 104.0], [1, 2], value_column="amount")

    return RelatedTables(
        tables={"users": users, "orders": orders},
        relationships=[
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )


def _target() -> TableTensor:
    return TableTensor.from_tensor(torch.tensor([[0.0], [1.0]]))


def test_table_dispatch_scopes_and_fits_related_tables_independently() -> None:
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    x_query = _table([3.0], [3], value_column="feature")
    related_context = _related_tables(query=False)
    related_query = _related_tables(query=True)
    recipe = sp.Recipe(
        features=sp.TableDispatch(
            related=sp.StypeDispatch(numerical=sp.Standardize()),
        ),
    )

    execution = _RecipeExecution._bind(
        recipe=recipe,
        x=x_context,
        y=_target(),
        related_tables=related_context,
        num_members=1,
        generator=None,
    )
    (query,) = execution.transform(x=x_query, related_tables=related_query)
    (context,) = execution.contexts

    assert context.related_tables is not None
    assert query.related_tables is not None
    torch.testing.assert_close(context.x.numerical, x_context.numerical)
    torch.testing.assert_close(query.x.numerical, x_query.numerical)
    for table in context.related_tables.tables.values():
        torch.testing.assert_close(
            table.numerical,
            torch.tensor([[-1.0], [1.0]]),
        )
    torch.testing.assert_close(
        query.related_tables.tables["users"].numerical,
        torch.tensor([[3.0]]),
    )
    torch.testing.assert_close(
        query.related_tables.tables["orders"].numerical,
        torch.tensor([[2.0]]),
    )


def test_nested_table_dispatch_routes_ensemble_members() -> None:
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    x_query = _table([3.0], [3], value_column="feature")
    related_context = _related_tables(query=False)
    related_query = _related_tables(query=True)
    recipe = sp.Recipe(
        features=sp.StypeDispatch(
            numerical=sp.Choice(
                sp.Sequential(
                    sp.TableDispatch(
                        task=_Add(10),
                        related=_Add(100),
                    ),
                ),
                _Add(1),
                selection="round_robin",
            ),
        ),
    )

    execution = _RecipeExecution._bind(
        recipe=recipe,
        x=x_context,
        y=_target(),
        related_tables=related_context,
        num_members=2,
        generator=None,
    )
    first, second = execution.transform(
        x=x_query,
        related_tables=related_query,
    )

    assert first.related_tables is not None
    assert second.related_tables is not None
    torch.testing.assert_close(
        first.x.numerical,
        x_query.numerical + 10,
    )
    torch.testing.assert_close(
        first.related_tables.tables["users"].numerical,
        related_query.tables["users"].numerical + 100,
    )
    torch.testing.assert_close(
        second.x.numerical,
        x_query.numerical + 1,
    )
    torch.testing.assert_close(
        second.related_tables.tables["users"].numerical,
        related_query.tables["users"].numerical + 1,
    )


def test_recipe_rejects_table_dispatch_outside_features() -> None:
    with pytest.raises(ValueError, match=r"Recipe\.features"):
        sp.Recipe(target=sp.TableDispatch(related=sp.Identity()))
    with pytest.raises(ValueError, match=r"Recipe\.features"):
        sp.Recipe(output=sp.TableDispatch(related=sp.Identity()))
