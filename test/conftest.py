import pandas as pd
import pytest
import torch
from sdm import RelationalData, TableTensor, infer_stypes


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def _torch_warn_always() -> None:
    torch.set_warn_always(True)


@pytest.fixture
def relational_data(device: torch.device) -> RelationalData:
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
                stypes=infer_stypes(
                    users_df,
                    overrides={"city": "categorical"},
                ),
            ),
            "orders": TableTensor.from_pandas(
                df=orders_df,
                stypes=infer_stypes(orders_df),
            ),
            "items": TableTensor.from_pandas(
                df=items_df,
                stypes=infer_stypes(
                    items_df,
                    overrides={"category": "categorical"},
                ),
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
    ).to(device)
