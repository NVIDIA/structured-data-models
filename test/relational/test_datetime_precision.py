# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pyarrow as pa
import pytest
import torch

from sdm import RelationalData, Relationship, TableTensor, TaskLink
from sdm.testing import withCUDA


def _data(event_times: pa.Array, parent_time: pa.Array) -> RelationalData:
    parents = TableTensor.from_arrow(
        pa.table({"parent_id": [0], "time": parent_time}),
        {"parent_id": "id", "time": "datetime"},
    )
    events = TableTensor.from_arrow(
        pa.table(
            {
                "parent_id": [0] * len(event_times),
                "event_id": range(len(event_times)),
                "time": event_times,
            }
        ),
        {"parent_id": "id", "event_id": "id", "time": "datetime"},
    )
    return RelationalData(
        tables={"parents": parents, "events": events},
        relationships=[
            Relationship(
                left_table="events",
                left_columns=("parent_id",),
                right_table="parents",
                right_columns=("parent_id",),
            )
        ],
    )


def _sample(
    data: RelationalData, times: pa.Array
) -> tuple[list[list[int]], TableTensor]:
    queries = TableTensor.from_arrow(
        pa.table({"parent_id": [0] * len(times), "time": times}),
        {"parent_id": "id", "time": "datetime"},
        device=data.device,
    )
    output = data.sampler(time_columns={"parents": "time", "events": "time"})(
        task_table=queries,
        task_link=TaskLink(
            task_columns=("parent_id",),
            table="parents",
            table_columns=("parent_id",),
        ),
        task_time_column="time",
        num_neighbors=[10],
        temporal_strategy="last",
    )
    frame = output.related_tables.tables["events"].to_pandas()
    groups = {
        int(i): sorted(values.event_id.tolist())
        for i, values in frame.groupby("__example__")
    }
    return [groups.get(i, []) for i in range(len(times))], output.task_table


@withCUDA
@pytest.mark.parametrize("base", [-1_000_000, 1_704_067_200_000_000_000])
def test_ns_cutoffs_preserve_anchor_observations_and_exclude_next_microsecond(
    base: int,
    device: torch.device,
) -> None:
    pytest.importorskip("pyg_lib")
    if device.type == "cuda":
        pytest.importorskip("pylibcugraph")
    data = _data(
        pa.array(
            [base // 1_000 - 1, base // 1_000, base // 1_000 + 1],
            type=pa.timestamp("us"),
        ),
        pa.array([base // 1_000 - 10], type=pa.timestamp("us")),
    )
    data = data.to(device)
    times = pa.array(
        [base - 1, base, base + 1, base + 999, base + 1_000, base + 1_001],
        type=pa.timestamp("ns"),
    )
    actual, returned_queries = _sample(data, times)
    assert actual == [[0], [0, 1], [0, 1], [0, 1], [0, 1, 2], [0, 1, 2]]
    assert returned_queries.datetime_units == ("ns",)
    assert (
        returned_queries["time"]
        .to_arrow()["time"]
        .combine_chunks()
        .equals(times)
    )
    assert data.tables["events"].datetime_units == ("us",)


@withCUDA
def test_ns_event_equality_and_ordering_with_mixed_event_clocks(
    device: torch.device,
) -> None:
    pytest.importorskip("pyg_lib")
    if device.type == "cuda":
        pytest.importorskip("pylibcugraph")
    base = 1_704_067_200_000_000_000
    data = _data(
        pa.array(
            [base - 1, base, base + 1, base + 1_000], type=pa.timestamp("ns")
        ),
        pa.array([base // 1_000 - 10], type=pa.timestamp("us")),
    )
    data = data.to(device)
    actual, _ = _sample(
        data,
        pa.array(
            [base - 1, base, base + 1, base + 2], type=pa.timestamp("ns")
        ),
    )
    assert actual == [[0], [0, 1], [0, 1, 2], [0, 1, 2]]
    # A microsecond query must not admit +1ns events in a nanosecond graph.
    actual, returned = _sample(
        data, pa.array([base // 1_000], type=pa.timestamp("us"))
    )
    assert actual == [[0, 1]]
    assert returned.datetime_units == ("us",)


def test_legacy_wide_us_graph_and_explicit_new_mixed_range_error() -> None:
    pytest.importorskip("pyg_lib")
    wide = pa.array([datetime(3000, 1, 1)], type=pa.timestamp("us"))
    data = _data(wide, wide)
    actual, _ = _sample(data, wide)
    assert actual == [[0]]
    mixed = _data(pa.array([1], type=pa.timestamp("ns")), wide)
    with pytest.raises(ValueError, match="nanosecond timestamp range"):
        mixed.sampler(time_columns={"parents": "time", "events": "time"})
    assert (
        data.tables["parents"].to_arrow()["time"].combine_chunks().equals(wide)
    )
