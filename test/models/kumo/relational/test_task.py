# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import torch

from sdm import RelatedTables, TableTensor
from sdm.models.kumo.relational.task import TaskGraph
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("num_hops", [0, 1, 2, 8, None])
def test_task_assignment(
    device: torch.device,
    num_hops: int | None,
) -> None:
    nodes = TableTensor.from_pandas(
        df=pd.DataFrame({"id": range(8), "parent": [1, 2, 0, 2, 5, 6, 4, 7]}),
        stypes={"id": "id", "parent": "id"},
        device=device,
    )
    task = TableTensor.from_pandas(
        df=pd.DataFrame({"id": [5, 3]}),
        stypes={"id": "id"},
        device=device,
    )
    relationship = {
        "left_table": "nodes",
        "left_column": "parent",
        "right_table": "nodes",
        "right_column": "id",
    }
    related_tables = RelatedTables(
        tables={"nodes": nodes},
        # Duplicate edges within two disjoint cyclic neighborhoods.
        relationships=[relationship, relationship],
        task_links=[
            {"task_column": "id", "table": "nodes", "table_column": "id"}
        ],
    )

    graph = TaskGraph.from_input(task, related_tables, num_hops=num_hops)

    expected = {
        0: [-1, -1, -1, 1, -1, 0, -1, -1],
        1: [-1, -1, 1, 1, 0, 0, 0, -1],
    }.get(num_hops, [1, 1, 1, 1, 0, 0, 0, -1])
    assert graph.task_row_by_table["nodes"].equal(
        torch.tensor(expected, device=device)
    )
    assert graph.readout_index.equal(torch.tensor([5, 3], device=device))
    assert graph.num_hops == (2 if num_hops is None else num_hops)


@withCUDA
@pytest.mark.parametrize("num_hops", [0, 3, None])
@pytest.mark.parametrize(
    ("num_nodes", "task_ids"), [(0, []), (3, []), (3, [1])]
)
def test_task_assignment_without_edges(
    device: torch.device,
    num_hops: int | None,
    num_nodes: int,
    task_ids: list[int],
) -> None:
    nodes = TableTensor.from_pandas(
        df=pd.DataFrame({"id": pd.Series(range(num_nodes), dtype="int64")}),
        stypes={"id": "id"},
        device=device,
    )
    task = TableTensor.from_pandas(
        df=pd.DataFrame({"id": pd.Series(task_ids, dtype="int64")}),
        stypes={"id": "id"},
        device=device,
    )
    related_tables = RelatedTables(
        tables={"nodes": nodes},
        relationships=[],
        task_links=[
            {"task_column": "id", "table": "nodes", "table_column": "id"}
        ],
    )

    graph = TaskGraph.from_input(task, related_tables, num_hops=num_hops)

    expected = torch.full((num_nodes,), -1, device=device)
    expected[task_ids] = torch.arange(len(task_ids), device=device)
    assert graph.task_row_by_table["nodes"].equal(expected)
    assert graph.num_hops == (0 if num_hops is None else num_hops)
