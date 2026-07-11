import pytest
import torch
from sdm import (
    ColumnarTensor,
    RelationalData,
    Relationship,
    Stype,
    TableTensor,
    TaskLink,
)


def test_sampler(data: RelationalData) -> None:
    pytest.importorskip("pyg_lib")

    task_table, related_tables = data.sampler()(
        task_table=TableTensor(
            columns={"id": ("user_id",)},
            id=ColumnarTensor((torch.tensor([3, 2, 1, 0]),)),
        ),
        task_link={
            "task_column": "user_id",
            "table": "users",
            "table_columns": "user_id",
        },
        num_neighbors=[10, 10],
    )

    assert task_table.columns[Stype.id] == ("user_id", "__example__")
    assert task_table.id[..., 0].equal(torch.tensor([3, 2, 1, 0]))
    assert task_table.id[..., 1].equal(torch.tensor([0, 1, 2, 3]))

    tables = related_tables.tables
    assert len(tables) == 3
    assert tables["users"].columns[Stype.id] == ("user_id", "__example__")
    assert tables["users"].id[..., 0].equal(torch.tensor([3, 2, 1, 0]))
    assert tables["users"].id[..., 1].equal(torch.tensor([0, 1, 2, 3]))
    assert tables["orders"].columns[Stype.id] == (
        "user_id",
        "item_id",
        "__example__",
    )
    assert tables["orders"].id[..., 0].equal(torch.tensor([3, 3, 3, 1, 0, 0]))
    assert tables["orders"].id[..., 1].tolist() == [
        "A",
        "B",
        "A",
        "C",
        "A",
        "B",
    ]
    assert tables["orders"].id[..., 2].equal(torch.tensor([0, 0, 0, 2, 3, 3]))
    assert tables["items"].columns[Stype.id] == ("item_id", "__example__")
    assert tables["items"].id[..., 0].tolist() == ["A", "B", "C", "A", "B"]
    assert tables["items"].id[..., 1].equal(torch.tensor([0, 0, 2, 3, 3]))

    assert related_tables.relationships == (
        Relationship(
            left_table="orders",
            left_columns=("__example__", "user_id"),
            right_table="users",
            right_columns=("__example__", "user_id"),
        ),
        Relationship(
            left_table="orders",
            left_columns=("__example__", "item_id"),
            right_table="items",
            right_columns=("__example__", "item_id"),
        ),
    )
    assert related_tables.task_links == (
        TaskLink(
            task_columns=("__example__", "user_id"),
            table="users",
            table_columns=("__example__", "user_id"),
        ),
    )
