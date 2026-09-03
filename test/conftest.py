# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch

from sdm import RelationalData, Stype, TableTensor, infer_stypes


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
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                    None,
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-07",
                ]
            ),
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
                stypes=infer_stypes(users_df, id="infer"),
            ),
            "orders": TableTensor.from_pandas(
                df=orders_df,
                stypes=infer_stypes(orders_df, id="infer"),
            ),
            "items": TableTensor.from_pandas(
                df=items_df,
                stypes=infer_stypes(items_df, id="infer"),
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


@pytest.fixture
def temporal_data() -> RelationalData:
    return RelationalData(
        tables={
            "roots": TableTensor.from_pandas(
                df=pd.DataFrame({"root_id": [0]}),
                stypes={"root_id": Stype.id},
            ),
            "first": TableTensor.from_pandas(
                df=pd.DataFrame(
                    {
                        "first_id": [10, 11, 12],
                        "root_id": [0, 0, 0],
                        "time": pd.to_datetime([3, 1, 2], unit="s"),
                    }
                ),
                stypes={
                    "first_id": Stype.id,
                    "root_id": Stype.id,
                    "time": Stype.datetime,
                },
            ),
            "second": TableTensor.from_pandas(
                df=pd.DataFrame(
                    {
                        "second_id": [20, 21],
                        "first_id": [10, 12],
                        "time": pd.to_datetime([8, 9], unit="s"),
                    }
                ),
                stypes={
                    "second_id": Stype.id,
                    "first_id": Stype.id,
                    "time": Stype.datetime,
                },
            ),
        },
        relationships=[
            {
                "left_table": "first",
                "left_column": "root_id",
                "right_table": "roots",
                "right_column": "root_id",
            },
            {
                "left_table": "second",
                "left_column": "first_id",
                "right_table": "first",
                "right_column": "first_id",
            },
        ],
    )
