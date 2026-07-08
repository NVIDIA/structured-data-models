from dataclasses import FrozenInstanceError

import pandas as pd
import pytest
import torch
from sdm import RelationalData, TableTensor, infer_stypes


@pytest.fixture
def data() -> RelationalData:
    users_df = pd.DataFrame(
        {
            "user_id": [0, 1, 2, 3],
            "age": [25, 30, 35, 40],
            "city": ["NYC", "LA", "Chicago", "NYC"],
        }
    )
    orders_df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 3, 3, 3],
            "item_id": ["A", "B", "C", "A", "B", "A"],
            "amount": [29.99, 49.99, 19.99, 99.99, 199.99, 39.99],
        }
    )
    items_df = pd.DataFrame(
        {
            "item_id": ["A", "B", "C"],
            "category": ["A", "B", "C"],
        }
    )

    return RelationalData(
        tables={
            "users": TableTensor.from_pandas(
                df=users_df,
                stypes=infer_stypes(users_df),
            ),
            "orders": TableTensor.from_pandas(
                df=orders_df,
                stypes=infer_stypes(orders_df),
            ),
            "items": TableTensor.from_pandas(
                df=items_df,
                stypes=infer_stypes(items_df),
            ),
        },
        relationships=[
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
        ],
    )


def test_edge_indices(data: RelationalData) -> None:
    edge_indices = data.edge_indices()

    assert len(edge_indices) == 2
    assert edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 3, 3]])
    )
    assert edge_indices[1].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 0]])
    )


def test_relationships_cannot_be_modified(data: RelationalData) -> None:
    num_relationships = len(data.relationships)

    with pytest.raises(FrozenInstanceError):
        data.relationships += data.relationships[:1]  # type: ignore

    assert len(data.relationships) == num_relationships


def test_homogeneous_graph(data: RelationalData) -> None:
    graph = data.homogeneous_graph()

    assert graph.edge_index.size() == (2, 24)
    assert graph.edge_index.dtype == torch.int64
    assert graph.node_offsets == {"users": 0, "orders": 4, "items": 10}
