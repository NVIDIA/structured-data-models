import pandas as pd
import pytest
import torch
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    RelatedTables,
    Relationship,
    TableTensor,
)


@pytest.fixture
def data() -> tuple[TableTensor, RelatedTables]:
    task_table = TableTensor(
        columns={
            "categorical": ["target"],
            "id": ["instance_id", "user_id"],
        },
        categorical=CategoricalTensor(
            data=torch.randint(0, 2, (4, 1)),
            categories=(torch.tensor([False, True]),),
        ),
        id=ColumnarTensor((torch.arange(4), torch.arange(4))),
    )

    users_df = pd.DataFrame(
        {
            "instance_id": [0, 1, 2, 3],
            "user_id": [0, 1, 2, 3],
            "age": [25, 30, 35, 40],
            "city": ["NYC", "LA", "Chicago", "NYC"],
        }
    )
    orders_df = pd.DataFrame(
        {
            "instance_id": [0, 0, 1, 3, 3, 3],
            "user_id": [0, 0, 1, 3, 3, 3],
            "item_id": ["A", "B", "C", "A", "B", "A"],
            "amount": [29.99, 49.99, 19.99, 99.99, 199.99, 39.99],
        }
    )
    items_df = pd.DataFrame(
        {
            "instance_id": [0, 0, 1, 3, 3],
            "item_id": ["A", "B", "C", "A", "B"],
            "category": ["A", "B", "C", "A", "B"],
        }
    )

    related_tables = RelatedTables(
        tables={
            "users": TableTensor.from_pandas(
                df=users_df,
                stypes={
                    "instance_id": "id",
                    "user_id": "id",
                    "age": "numerical",
                    "city": "categorical",
                },
            ),
            "orders": TableTensor.from_pandas(
                df=orders_df,
                stypes={
                    "instance_id": "id",
                    "user_id": "id",
                    "item_id": "id",
                    "amount": "numerical",
                },
            ),
            "items": TableTensor.from_pandas(
                df=items_df,
                stypes={
                    "instance_id": "id",
                    "item_id": "id",
                    "category": "categorical",
                },
            ),
        },
        relationships=[
            {
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            },
            {
                "left_table": "orders",
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            },
            {
                "left_table": "orders",
                "left_columns": ["instance_id", "item_id"],
                "right_table": "items",
                "right_columns": ["instance_id", "item_id"],
            },
        ],
    )

    return task_table, related_tables


def test_relationship_inputs_are_snapshotted(
    data: tuple[TableTensor, RelatedTables],
) -> None:
    _, related_tables = data
    left_columns = ["instance_id"]
    right_columns = ["instance_id"]
    relationship = Relationship(
        left_table=None,
        left_columns=left_columns,
        right_table="users",
        right_columns=right_columns,
    )
    relationships = [relationship]
    related_tables = RelatedTables(
        tables={"users": related_tables.tables["users"]},
        relationships=relationships,
    )

    left_columns.append("user_id")
    right_columns.append("user_id")
    relationships.clear()

    assert relationship.left_columns == ("instance_id",)
    assert relationship.right_columns == ("instance_id",)
    assert related_tables.relationships == (relationship,)


@pytest.mark.parametrize("task_on_left", [True, False])
def test_task_relationship_columns_must_be_id(
    data: tuple[TableTensor, RelatedTables],
    task_on_left: bool,
) -> None:
    task_table, related_tables = data
    if task_on_left:
        relationship = Relationship(
            left_table=None,
            left_columns=("target",),
            right_table="users",
            right_columns=("user_id",),
        )
    else:
        relationship = Relationship(
            left_table="users",
            left_columns=("user_id",),
            right_table=None,
            right_columns=("target",),
        )
    related_tables = RelatedTables(
        tables={"users": related_tables.tables["users"]},
        relationships=(relationship,),
    )

    match = "column 'target' in task table.*semantic type 'id'"
    with pytest.raises(ValueError, match=match):
        related_tables.edge_indices(task_table)
    with pytest.raises(ValueError, match=match):
        related_tables.homogeneous_graph(task_table)


def test_edge_indices(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data
    edge_indices = related_tables.edge_indices(task_table)

    assert len(edge_indices) == 3
    assert edge_indices[0].equal(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert edge_indices[1].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 3, 3]])
    )
    assert edge_indices[2].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 3]])
    )


def test_homogeneous_graph(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data
    graph = related_tables.homogeneous_graph(task_table)

    assert graph.num_rows == 15
    assert graph.num_task_rows == 4
    assert graph.node_offsets == {"users": 0, "orders": 4, "items": 10}
    assert graph.edge_index.size() == (2, 24)
    assert graph.edge_index.dtype == torch.int64
    forward_edge_index = torch.tensor([[4, 5, 6, 7, 8, 9], [0, 0, 1, 3, 3, 3]])
    assert graph.edge_index[:, :6].equal(forward_edge_index)
    assert graph.edge_index[:, 6:12].equal(forward_edge_index.flip(0))
    assert graph.edge_type.equal(torch.arange(4).repeat_interleave(6))
    assert graph.edge_type.dtype == torch.long
    assert graph.num_edge_types == 4
    assert graph.node_batch.equal(
        torch.tensor([0, 1, 2, 3, 0, 0, 1, 3, 3, 3, 0, 0, 1, 3, 3])
    )
    assert graph.num_hops == 2
    assert set(graph.task_edge_indices) == {0}
    assert graph.task_edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    )
    assert graph.task_edge_indices[0].dtype == torch.long


def test_empty_relationship_reserves_edge_types(
    data: tuple[TableTensor, RelatedTables],
) -> None:
    task_table, related_tables = data
    missing = TableTensor.from_pandas(
        df=pd.DataFrame({"instance_id": [99]}),
        stypes={"instance_id": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "users": related_tables.tables["users"],
            "orders": related_tables.tables["orders"],
            "missing": missing,
        },
        relationships=[
            {
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            },
            {
                "left_table": "orders",
                "left_column": "instance_id",
                "right_table": "missing",
                "right_column": "instance_id",
            },
            {
                "left_table": "orders",
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            },
        ],
    )

    graph = related_tables.homogeneous_graph(task_table)

    assert graph.edge_index.size() == (2, 12)
    assert graph.edge_type.equal(torch.tensor([2] * 6 + [3] * 6))
    assert graph.num_edge_types == 4


def test_all_empty_relationships_preserve_edge_type_schema(
    data: tuple[TableTensor, RelatedTables],
) -> None:
    task_table, related_tables = data
    missing = TableTensor.from_pandas(
        df=pd.DataFrame({"id": [99]}),
        stypes={"id": "id"},
    )
    related_tables = RelatedTables(
        tables={
            "users": related_tables.tables["users"],
            "orders": related_tables.tables["orders"],
            "missing": missing,
        },
        relationships=[
            {
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            },
            {
                "left_table": "orders",
                "left_column": "instance_id",
                "right_table": "missing",
                "right_column": "id",
            },
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "missing",
                "right_column": "id",
            },
        ],
    )

    graph = related_tables.homogeneous_graph(task_table)

    assert graph.edge_index.size() == (2, 0)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_type.size() == (0,)
    assert graph.edge_type.dtype == torch.long
    assert graph.num_edge_types == 4


def test_task_edge_indices_are_global_with_task_on_left(
    data: tuple[TableTensor, RelatedTables],
) -> None:
    task_table, related_tables = data
    related_tables = RelatedTables(
        tables={
            "orders": related_tables.tables["orders"],
            "users": related_tables.tables["users"],
        },
        relationships=[
            {
                "left_columns": ["instance_id", "user_id"],
                "right_table": "users",
                "right_columns": ["instance_id", "user_id"],
            }
        ],
    )
    graph = related_tables.homogeneous_graph(task_table)

    assert graph.task_edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3], [6, 7, 8, 9]])
    )


def test_task_table_on_right(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data
    related_tables = RelatedTables(
        tables={
            "orders": related_tables.tables["orders"],
            "users": related_tables.tables["users"],
        },
        relationships=[
            {
                "left_table": "users",
                "left_columns": ["instance_id", "user_id"],
                "right_columns": ["instance_id", "user_id"],
            }
        ],
    )

    graph = related_tables.homogeneous_graph(task_table)

    assert graph.edge_index.size() == (2, 0)
    assert graph.edge_index.dtype == torch.long
    assert graph.edge_type.size() == (0,)
    assert graph.num_edge_types == 0
    assert graph.task_edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3], [6, 7, 8, 9]])
    )

    row_batch, num_hops = related_tables.row_batch(
        graph,
        return_num_hops=True,
    )
    assert row_batch.equal(torch.tensor([-1, -1, -1, -1, -1, -1, 0, 1, 2, 3]))
    assert num_hops == 0


def test_singular_column_keys(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data
    users = related_tables.tables["users"]
    related_tables = RelatedTables(
        tables={"users": users},
        relationships=[
            {
                "left_table": None,
                "left_column": "instance_id",
                "right_table": "users",
                "right_column": "instance_id",
            }
        ],
    )

    edge_indices = related_tables.edge_indices(task_table)

    assert edge_indices[0].equal(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))


def test_row_batch(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data
    graph = related_tables.homogeneous_graph(task_table)

    row_batch, num_rows = related_tables.row_batch(graph, return_num_hops=True)
    assert row_batch.equal(
        torch.tensor([0, 1, 2, 3, 0, 0, 1, 3, 3, 3, 0, 0, 1, 3, 3])
    )
    assert num_rows == 2


def test_row_batch_rejects_conflicting_root_assignments() -> None:
    task_table = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [0, 0]}),
        stypes={"user_id": "id"},
    )
    users = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [0]}),
        stypes={"user_id": "id"},
    )
    related_tables = RelatedTables(
        tables={"users": users},
        relationships=[
            {
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
    )
    with pytest.raises(ValueError, match="Conflicting task assignments"):
        related_tables.homogeneous_graph(task_table)


def test_row_batch_rejects_conflicting_propagated_assignments() -> None:
    task_table = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [0, 1]}),
        stypes={"user_id": "id"},
    )
    users = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [0, 1], "group_id": [7, 7]}),
        stypes={"user_id": "id", "group_id": "id"},
    )
    groups = TableTensor.from_pandas(
        df=pd.DataFrame({"group_id": [7]}),
        stypes={"group_id": "id"},
    )
    related_tables = RelatedTables(
        tables={"users": users, "groups": groups},
        relationships=[
            {
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            },
            {
                "left_table": "users",
                "left_column": "group_id",
                "right_table": "groups",
                "right_column": "group_id",
            },
        ],
    )
    with pytest.raises(ValueError, match="Conflicting task assignments"):
        related_tables.homogeneous_graph(task_table)


def test_row_batch_allows_duplicate_edges_with_same_task(
    data: tuple[TableTensor, RelatedTables],
) -> None:
    task_table, related_tables = data
    task_relationship = {
        "left_columns": ["instance_id", "user_id"],
        "right_table": "users",
        "right_columns": ["instance_id", "user_id"],
    }
    table_relationship = {
        "left_table": "orders",
        "left_columns": ["instance_id", "user_id"],
        "right_table": "users",
        "right_columns": ["instance_id", "user_id"],
    }
    related_tables = RelatedTables(
        tables={
            "users": related_tables.tables["users"],
            "orders": related_tables.tables["orders"],
        },
        relationships=[
            task_relationship,
            task_relationship,
            table_relationship,
            table_relationship,
        ],
    )
    graph = related_tables.homogeneous_graph(task_table)

    row_batch, num_hops = related_tables.row_batch(
        graph,
        return_num_hops=True,
    )
    assert row_batch.equal(torch.tensor([0, 1, 2, 3, 0, 0, 1, 3, 3, 3]))
    assert num_hops == 1
