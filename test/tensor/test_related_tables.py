import pandas as pd
import pytest
import torch
from sdm import CategoricalTensor, ColumnarTensor, RelatedTables, TableTensor


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


def test_node_batch(data: tuple[TableTensor, RelatedTables]) -> None:
    task_table, related_tables = data

    out = related_tables.node_batch(task_table)

    assert out.equal(
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 3, 3, 3, 0, 0, 1, 3, 3])
    )
