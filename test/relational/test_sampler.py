# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent
from typing import Any, cast

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
from sdm.models import TabICLv2
from sdm.processing.execution import RecipeExecution


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
              size=(6, 5),
              blocks={
                numerical (1): [amount],
                datetime (1): [timestamp],
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


def test_batch_sampler(relational_data: RelationalData) -> None:
    pytest.importorskip("pyg_lib")

    task_table, related_tables = relational_data.sampler()(
        task_table=TableTensor(
            columns={"id": ("user_id",)},
            id=ColumnarTensor((torch.tensor([[3, 2], [0, 1]]),)),
        ),
        task_link={
            "task_column": "user_id",
            "table": "users",
            "table_columns": "user_id",
        },
        num_neighbors=[2, 2],
    )

    assert task_table.columns[Stype.id] == ("user_id", "__example__")
    assert task_table.id[..., 0].equal(torch.tensor([[3, 2], [0, 1]]))
    assert task_table.id[..., 1].equal(torch.tensor([[0, 1], [0, 1]]))

    assert len(related_tables.tables) == 3
    assert all(t.num_groups == 2 for t in related_tables.tables.values())
    assert all(t.num_members == 2 for t in related_tables.tables.values())

    users = related_tables.tables["users"]
    for member_id, expected in enumerate(([3, 2], [0, 1])):
        user = users.table(member_id)
        assert user.columns[Stype.id] == ("user_id", "__example__")
        assert user.id[..., 0].equal(torch.tensor(expected))
        assert user.id[..., 1].equal(torch.tensor([0, 1]))


def test_expanded_sampler(relational_data: RelationalData) -> None:
    pytest.importorskip("pyg_lib")

    task_table = TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.tensor([3, 2]),)),
    )
    task_table = cast(
        TableTensor,
        task_table.unsqueeze(0).expand(2, -1, -1),
    )

    task_table, related_tables = relational_data.sampler()(
        task_table=task_table,
        task_link={
            "task_column": "user_id",
            "table": "users",
            "table_columns": "user_id",
        },
        num_neighbors=[2, 2],
    )

    assert all(
        all(column.stride(0) == 0 for column in block.unbind(-1))
        if isinstance(block, ColumnarTensor)
        else block.stride(0) == 0
        for _, block in task_table.items()
        if block.size(-1) > 0
    )

    assert task_table.columns[Stype.id] == ("user_id", "__example__")
    assert task_table.id[..., 0].equal(torch.tensor([[3, 2], [3, 2]]))
    assert task_table.id[..., 1].equal(torch.tensor([[0, 1], [0, 1]]))

    assert len(related_tables.tables) == 3
    assert all(t.num_groups == 2 for t in related_tables.tables.values())
    assert all(t.num_members == 2 for t in related_tables.tables.values())

    users = related_tables.tables["users"]
    assert len({user.numerical.data_ptr() for user in users}) == 1
    for user in users:
        user = user.squeeze(0)
        assert user.columns[Stype.id] == ("user_id", "__example__")
        assert user.id[..., 0].equal(torch.tensor([3, 2]))
        assert user.id[..., 1].equal(torch.tensor([0, 1]))


def test_batched_recipe_execution(relational_data: RelationalData) -> None:
    pytest.importorskip("pyg_lib")

    context = TableTensor(
        columns={"numerical": ("target",), "id": ("user_id",)},
        numerical=torch.randn(2, 2, 1),
        id=ColumnarTensor((torch.tensor([[3, 2], [0, 1]]),)),
    )
    query = cast(TableTensor, context[0].expand(2, *context.size()[1:]))

    kwargs: dict[str, Any] = {
        "task_link": {
            "task_column": "user_id",
            "table": "users",
            "table_columns": "user_id",
        },
        "num_neighbors": [2, 2],
    }

    sampler = relational_data.sampler()
    context, related_context_tables = sampler(context, **kwargs)
    query, related_query_tables = sampler(query, **kwargs)

    recipe_execution = RecipeExecution(TabICLv2.default_recipe())
    contexts = recipe_execution.fit_transform(
        x=context.drop_columns("target"),
        y=context["target"],
        related_tables=related_context_tables,
    )
    queries = recipe_execution.transform(
        x=query.drop_columns("target"),
        related_tables=related_query_tables,
    )
    assert len(contexts) == 2
    assert len(queries) == 2
