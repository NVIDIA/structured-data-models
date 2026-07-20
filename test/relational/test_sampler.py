from textwrap import dedent

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


def test_sampler(relational_data: RelationalData) -> None:
    pytest.importorskip("pyg_lib")

    task_table, related_tables = relational_data.sampler()(
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
            left_columns=("user_id", "__example__"),
            right_table="users",
            right_columns=("user_id", "__example__"),
        ),
        Relationship(
            left_table="orders",
            left_columns=("item_id", "__example__"),
            right_table="items",
            right_columns=("item_id", "__example__"),
        ),
    )
    assert related_tables.task_links == (
        TaskLink(
            task_columns=("user_id", "__example__"),
            table="users",
            table_columns=("user_id", "__example__"),
        ),
    )

    assert repr(related_tables) == dedent("""\
        RelatedTables(
          tables={
            users: TableTensor(
              size=(4, 4),
              blocks={
                numerical (1): [age],
                categorical (1): [city],
                id (2): [user_id, __example__],
              },
            ),
            orders: TableTensor(
              size=(6, 4),
              blocks={
                numerical (1): [amount],
                id (3): [user_id, item_id, __example__],
              },
            ),
            items: TableTensor(
              size=(5, 3),
              blocks={
                categorical (1): [category],
                id (2): [item_id, __example__],
              },
            ),
          },
          relationships=[
            orders.[user_id,__example__] <> users.[user_id,__example__],
            orders.[item_id,__example__] <> items.[item_id,__example__],
          ],
          task_links=[
            [user_id,__example__] > users.[user_id,__example__],
          ],
        )""")

    edge_indices, task_links = related_tables.edge_indices(task_table)
    assert len(edge_indices) == 2
    assert edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 2, 3, 3]])
    )
    assert edge_indices[1].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 0, 2, 3, 4]])
    )
    assert len(task_links) == 1
    assert task_links[0].equal(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
