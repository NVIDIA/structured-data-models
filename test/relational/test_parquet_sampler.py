# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import sdm

pytest.importorskip("polars")
pytest.importorskip("pyg_lib")


def test_parquet_sampler_matches_in_memory_sampler(tmp_path: Path) -> None:
    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "age": [20.0, 30.0, 40.0],
            "tier": ["gold", "silver", "bronze"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": [10, 11, 12, 13],
            "user_id": [0, 0, 1, 1],
            "amount": [1.0, 2.0, 3.0, 4.0],
            "time": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"]
            ),
        }
    )
    frames = {"users": users, "orders": orders}
    stypes = {
        "users": {
            "user_id": "id",
            "age": "numerical",
            "tier": "categorical",
        },
        "orders": {
            "order_id": "id",
            "user_id": "id",
            "amount": "numerical",
            "time": "datetime",
        },
    }
    paths = {name: tmp_path / f"{name}.parquet" for name in frames}
    for name, frame in frames.items():
        frame.to_parquet(paths[name], index=False)

    relationships = [
        {
            "left_table": "orders",
            "left_column": "user_id",
            "right_table": "users",
            "right_column": "user_id",
        }
    ]
    time_columns = {"orders": "time"}
    memory = sdm.RelationalData(
        tables={
            name: sdm.TableTensor.from_pandas(frame, stypes[name])
            for name, frame in frames.items()
        },
        relationships=relationships,
    ).sampler(time_columns=time_columns)
    disk = sdm.ParquetRelationalSampler(
        tables=paths,
        stypes=stypes,
        relationships=relationships,
        time_columns=time_columns,
    )
    task = sdm.TableTensor.from_pandas(
        pd.DataFrame(
            {
                "user_id": [0, 1],
                "time": pd.to_datetime(["2020-01-04", "2020-01-05"]),
            }
        ),
        {"user_id": "id", "time": "datetime"},
    )
    kwargs: dict[str, Any] = {
        "task_link": {
            "task_column": "user_id",
            "table": "users",
            "table_column": "user_id",
        },
        "num_neighbors": [2],
        "task_time_column": "time",
    }

    expected = memory(task, **kwargs)
    actual = disk(task, **kwargs)

    pd.testing.assert_frame_equal(
        actual.task_table.to_pandas(), expected.task_table.to_pandas()
    )
    actual_related = actual.related_tables
    expected_related = expected.related_tables
    assert actual_related.relationships == expected_related.relationships
    assert actual_related.task_links == expected_related.task_links
    assert actual_related.tables.keys() == expected_related.tables.keys()
    for name, table in actual.related_tables.tables.items():
        pd.testing.assert_frame_equal(
            table.to_pandas(), expected_related.tables[name].to_pandas()
        )

    orders.loc[0, "amount"] = 99.0
    orders.to_parquet(paths["orders"], index=False)
    refreshed = disk(task, **kwargs)
    assert (
        99.0
        in refreshed.related_tables.tables["orders"]
        .to_pandas()["amount"]
        .tolist()
    )
