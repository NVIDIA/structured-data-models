# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd
import pytest

from sdm import RelationalData, Stype, TableTensor
from sdm.relational.sampler import EXAMPLE_ID, RelationalSamplerOutput
from sdm.testing import onlyCUDA


def _require_backends() -> None:
    pytest.importorskip("pyg_lib")
    pytest.importorskip("cudf")
    pytest.importorskip("pylibcugraph")


def _table(
    data: dict[str, list[Any]],
    stypes: dict[str, Stype],
) -> TableTensor:
    return TableTensor.from_pandas(df=pd.DataFrame(data), stypes=stypes)


def _cuda_table(table: TableTensor) -> TableTensor:
    return cast(TableTensor, table.cuda())


def _sample(
    data: RelationalData,
    task_table: TableTensor,
    task_link: Mapping[str, str | Sequence[str]],
    num_neighbors: Sequence[int],
) -> RelationalSamplerOutput:
    return data.sampler()(
        task_table=task_table,
        task_link=task_link,
        num_neighbors=num_neighbors,
    )


def _canonical_table(
    table: TableTensor,
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    arrow = table.to_arrow()
    columns = tuple(arrow.column_names)
    rows = tuple(
        tuple(row[column] for column in columns) for row in arrow.to_pylist()
    )
    return columns, tuple(sorted(rows, key=repr))


def _canonical_output(output: RelationalSamplerOutput) -> dict[str, Any]:
    output = output.cpu()
    related = output.related_tables
    return {
        "task_table": _canonical_table(output.task_table),
        "tables": tuple(
            (name, _canonical_table(table))
            for name, table in sorted(related.tables.items())
        ),
        "relationships": related.relationships,
        "task_links": related.task_links,
    }


def _related_rows(
    output: RelationalSamplerOutput,
    table_name: str,
) -> tuple[tuple[Any, ...], ...]:
    output = output.cpu()
    return _canonical_table(output.related_tables.tables[table_name])[1]


@onlyCUDA
@pytest.mark.parametrize(
    "num_neighbors",
    [[0], [-1], [-1, -1]],
    ids=("seed-only", "one-hop", "two-hop"),
)
def test_pyg_and_cugraph_match_non_temporal_samples(
    relational_data: RelationalData,
    num_neighbors: list[int],
) -> None:
    _require_backends()
    task_table = _table({"entity": [0, 2, 0]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }

    expected = _sample(relational_data, task_table, task_link, num_neighbors)
    actual = _sample(
        relational_data.cuda(),
        _cuda_table(task_table),
        task_link,
        num_neighbors,
    )

    assert _canonical_output(actual) == _canonical_output(expected)


@onlyCUDA
def test_pyg_and_cugraph_match_reverse_samples(
    relational_data: RelationalData,
) -> None:
    _require_backends()
    task_table = _table({"entity": ["A", "C"]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "items",
        "table_column": "item_id",
    }

    expected = _sample(relational_data, task_table, task_link, [-1, -1])
    actual = _sample(
        relational_data.cuda(),
        _cuda_table(task_table),
        task_link,
        [-1, -1],
    )

    assert _canonical_output(actual) == _canonical_output(expected)


@onlyCUDA
def test_pyg_and_cugraph_match_composite_seed_samples() -> None:
    _require_backends()
    data = RelationalData(
        tables={
            "users": _table(
                {
                    "account_id": [1, 2, 3],
                    "region": ["europe", "tokyo", "america"],
                },
                {"account_id": Stype.id, "region": Stype.id},
            ),
            "orders": _table(
                {
                    "order_id": [10, 11, 12],
                    "account_id": [1, 2, 3],
                    "region": ["europe", "tokyo", "america"],
                },
                {
                    "order_id": Stype.id,
                    "account_id": Stype.id,
                    "region": Stype.id,
                },
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_columns": ("account_id", "region"),
                "right_table": "users",
                "right_columns": ("account_id", "region"),
            }
        ],
    )
    task_table = _table(
        {
            "requested_account": [3, 1, 3],
            "requested_region": ["america", "europe", "america"],
        },
        {
            "requested_account": Stype.id,
            "requested_region": Stype.id,
        },
    )
    task_link = {
        "task_columns": ("requested_account", "requested_region"),
        "table": "users",
        "table_columns": ("account_id", "region"),
    }

    expected = _sample(data, task_table, task_link, [0])
    actual = _sample(
        data.cuda(),
        _cuda_table(task_table),
        task_link,
        [0],
    )

    expected_associations = (
        (0, 3, "america"),
        (1, 1, "europe"),
        (2, 3, "america"),
    )
    for output in (expected, actual):
        users = output.cpu().related_tables.tables["users"].to_arrow()
        values = users.select((EXAMPLE_ID, "account_id", "region")).to_pydict()
        associations = tuple(
            zip(
                values[EXAMPLE_ID],
                values["account_id"],
                values["region"],
            )
        )
        assert associations == expected_associations


@onlyCUDA
@pytest.mark.parametrize(
    ("source", "task"),
    [
        (([1, 2], ["a", "b"]), ([9], ["missing"])),
        (([1, 1], ["a", "a"]), ([1], ["a"])),
    ],
    ids=("missing", "duplicate"),
)
def test_pyg_and_cugraph_reject_invalid_composite_seed_matches(
    source: tuple[list[int], list[str]],
    task: tuple[list[int], list[str]],
) -> None:
    _require_backends()
    data = RelationalData(
        tables={
            "users": _table(
                {"account": source[0], "region": source[1]},
                {"account": Stype.id, "region": Stype.id},
            )
        },
        relationships=[],
    )
    task_table = _table(
        {"account": task[0], "region": task[1]},
        {"account": Stype.id, "region": Stype.id},
    )
    task_link = {
        "task_columns": ("account", "region"),
        "table": "users",
        "table_columns": ("account", "region"),
    }

    with pytest.raises(ValueError, match="match exactly one row"):
        _sample(data, task_table, task_link, [0])
    with pytest.raises(ValueError, match="match exactly one row"):
        _sample(data.cuda(), _cuda_table(task_table), task_link, [0])


@onlyCUDA
def test_pyg_and_cugraph_finite_fanout_satisfies_same_invariants(
    relational_data: RelationalData,
) -> None:
    _require_backends()
    task_table = _table({"entity": [0]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }

    cpu_all = _sample(relational_data, task_table, task_link, [-1])
    cpu_sample = _sample(relational_data, task_table, task_link, [1])
    gpu_data = relational_data.cuda()
    gpu_task = _cuda_table(task_table)
    gpu_all = _sample(gpu_data, gpu_task, task_link, [-1])
    gpu_sample = _sample(gpu_data, gpu_task, task_link, [1])

    assert _canonical_output(gpu_all) == _canonical_output(cpu_all)
    for sample, exhaustive in (
        (cpu_sample, cpu_all),
        (gpu_sample, gpu_all),
    ):
        rows = _related_rows(sample, "orders")
        assert len(rows) == 1
        assert set(rows) <= set(_related_rows(exhaustive, "orders"))
        assert _related_rows(sample, "users") == _related_rows(
            exhaustive,
            "users",
        )
