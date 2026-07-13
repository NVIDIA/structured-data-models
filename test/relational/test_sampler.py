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

    assert task_table.columns[Stype.id] == ("__example__", "user_id")
    assert task_table.id[..., 0].equal(torch.tensor([0, 1, 2, 3]))
    assert task_table.id[..., 1].equal(torch.tensor([3, 2, 1, 0]))

    tables = related_tables.tables
    assert len(tables) == 3
    assert tables["users"].columns[Stype.id] == ("__example__", "user_id")
    assert tables["users"].id[..., 0].equal(torch.tensor([0, 1, 2, 3]))
    assert tables["users"].id[..., 1].equal(torch.tensor([3, 2, 1, 0]))
    assert tables["orders"].columns[Stype.id] == (
        "__example__",
        "user_id",
        "item_id",
    )
    assert tables["orders"].id[..., 0].equal(torch.tensor([0, 0, 0, 2, 3, 3]))
    assert tables["orders"].id[..., 1].equal(torch.tensor([3, 3, 3, 1, 0, 0]))
    assert tables["orders"].id[..., 2].tolist() == [
        "A",
        "B",
        "A",
        "C",
        "A",
        "B",
    ]
    assert tables["items"].columns[Stype.id] == ("__example__", "item_id")
    assert tables["items"].id[..., 0].equal(torch.tensor([0, 0, 2, 3, 3]))
    assert tables["items"].id[..., 1].tolist() == ["A", "B", "C", "A", "B"]

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

    assert repr(related_tables) == dedent("""\
        RelatedTables(
          tables={
            users: TableTensor(
              size=(4, 4),
              blocks={
                numerical (1): [age],
                categorical (1): [city],
                id (2): [__example__, user_id],
              },
            ),
            orders: TableTensor(
              size=(6, 4),
              blocks={
                numerical (1): [amount],
                id (3): [__example__, user_id, item_id],
              },
            ),
            items: TableTensor(
              size=(5, 3),
              blocks={
                categorical (1): [category],
                id (2): [__example__, item_id],
              },
            ),
          },
          relationships=[
            orders.[__example__,user_id] <> users.[__example__,user_id],
            orders.[__example__,item_id] <> items.[__example__,item_id],
          ],
          task_links=[
            [__example__,user_id] -> users.[__example__,user_id],
          ],
        )""")
