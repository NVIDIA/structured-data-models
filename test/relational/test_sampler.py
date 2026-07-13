from textwrap import dedent

import pandas as pd
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
from sdm.relational.sampler import _coalesce_sampled_edges


def _table(df: pd.DataFrame, stypes: dict[str, str]) -> TableTensor:
    return TableTensor.from_pandas(df=df, stypes=stypes)


def _triangle_data() -> tuple[RelationalData, TableTensor]:
    data = RelationalData(
        tables={
            "users": _table(
                pd.DataFrame({"user_id": [0], "value": [1.0]}),
                {"user_id": "id", "value": "numerical"},
            ),
            "orders": _table(
                pd.DataFrame(
                    {
                        "order_id": [10],
                        "user_id": [0],
                        "item_id": [100],
                        "value": [2.0],
                    }
                ),
                {
                    "order_id": "id",
                    "user_id": "id",
                    "item_id": "id",
                    "value": "numerical",
                },
            ),
            "items": _table(
                pd.DataFrame(
                    {
                        "item_id": [100, 101],
                        "owner_id": [0, 0],
                        "time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                        "value": [3.0, 4.0],
                    }
                ),
                {
                    "item_id": "id",
                    "owner_id": "id",
                    "time": "datetime",
                    "value": "numerical",
                },
            ),
        },
        relationships=(
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            },
            {
                "left_table": "orders",
                "left_column": "item_id",
                "right_table": "items",
                "right_column": "item_id",
            },
            {
                "left_table": "items",
                "left_column": "owner_id",
                "right_table": "users",
                "right_column": "user_id",
            },
        ),
    )
    task_table = _table(
        pd.DataFrame(
            {
                "user_id": [0],
                "time": pd.to_datetime(["2024-01-03"]),
            }
        ),
        {"user_id": "id", "time": "datetime"},
    )
    return data, task_table


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
        num_neighbors=[-1, -1],
    )

    assert task_table.columns[Stype.id] == ("user_id", "__example__")
    assert task_table.id[..., 0].equal(torch.tensor([3, 2, 1, 0]))
    assert task_table.id[..., 1].equal(torch.tensor([0, 1, 2, 3]))

    tables = related_tables.tables
    assert related_tables.sample is not None
    assert related_tables.sample.num_neighbors == (-1, -1)
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


def test_restricted_fanout_preserves_exact_context() -> None:
    pytest.importorskip("pyg_lib")
    data, task_table = _triangle_data()

    sampled_task, related_tables = data.sampler(
        time_columns={"items": "time"}
    )(
        task_table=task_table,
        task_link={
            "task_column": "user_id",
            "table": "users",
            "table_column": "user_id",
        },
        num_neighbors=[1, 1],
        task_time_column="time",
    )

    sample = related_tables.sample
    assert sample is not None
    assert sample.num_hops == 2
    assert sample.num_neighbors == (1, 1)
    assert sample.disjoint
    assert sample.temporal
    assert sample.temporal_strategy == "last"
    assert sample.node_hops["users"].tolist() == [0]
    assert sample.node_hops["orders"].tolist() == [1]
    assert sample.node_hops["items"].tolist() == [1, 2]
    assert sample.task_edge_indices[0].tolist() == [[0], [0]]
    assert sample.seed_time is not None
    assert torch.equal(
        sample.seed_time,
        sampled_task["time"].datetime.squeeze(-1),
    )

    # Both item rows remain in the sampled node set, and both can join to the
    # root user. Only item row zero was actually traversed through this
    # relationship, so row one must not be restored by a relational join.
    assert sample.edge_indices[2].tolist() == [[0], [0]]
    assert sample.edge_hops[2].tolist() == [1]
    full_edges = RelationalData(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
    ).edge_indices()
    assert full_edges[2].tolist() == [[0, 1], [0, 0]]
    exact_edges, _ = related_tables.edge_indices(sampled_task)
    assert exact_edges[2].tolist() == [[0], [0]]

    moved = related_tables.to("cpu")
    assert moved.sample is not None
    assert moved.sample.edge_indices[2].tolist() == [[0], [0]]
    assert moved.sample.seed_time is not None
    assert torch.equal(moved.sample.seed_time, sample.seed_time)


def test_repeated_seeds_have_distinct_roots_and_preserve_empty_hops(
    data: RelationalData,
) -> None:
    pytest.importorskip("pyg_lib")
    task_table = TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.tensor([3, 3]),)),
    )

    _, related_tables = data.sampler()(
        task_table=task_table,
        task_link={
            "task_column": "user_id",
            "table": "users",
            "table_column": "user_id",
        },
        num_neighbors=[0, 0, 0],
    )

    sample = related_tables.sample
    assert sample is not None
    assert sample.num_hops == 3
    assert sample.num_neighbors == (0, 0, 0)
    assert sample.node_batch["users"].tolist() == [0, 1]
    assert sample.node_hops["users"].tolist() == [0, 0]
    assert sample.task_edge_indices[0].tolist() == [[0, 1], [0, 1]]
    assert related_tables.tables["users"].id[..., 0].tolist() == [3, 3]
    assert related_tables.tables["orders"].size(0) == 0
    assert related_tables.tables["items"].size(0) == 0
    assert all(edge_index.numel() == 0 for edge_index in sample.edge_indices)


def test_coalesce_sampled_edges_uses_earliest_discovery_hop() -> None:
    edge_index, edge_hop = _coalesce_sampled_edges(
        edge_index=torch.tensor([[1, 0, 1], [0, 1, 0]]),
        edge_hop=torch.tensor([2, 1, 1]),
        num_dst_nodes=2,
    )

    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert edge_hop.tolist() == [1, 1]
