from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd
import pytest
from sdm import RelationalData, Stype, TableTensor
from sdm.relational.sampler import RelationalSamplerOutput
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
    *,
    time_columns: Mapping[str, str] | None = None,
    task_time_column: str | None = None,
) -> RelationalSamplerOutput:
    return data.sampler(time_columns=time_columns)(
        task_table=task_table,
        task_link=task_link,
        num_neighbors=num_neighbors,
        task_time_column=task_time_column,
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
    data: RelationalData,
    num_neighbors: list[int],
) -> None:
    _require_backends()
    task_table = _table({"entity": [0, 2, 0]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }

    expected = _sample(data, task_table, task_link, num_neighbors)
    actual = _sample(
        data.cuda(),
        _cuda_table(task_table),
        task_link,
        num_neighbors,
    )

    assert _canonical_output(actual) == _canonical_output(expected)


@onlyCUDA
def test_pyg_and_cugraph_match_reverse_samples(data: RelationalData) -> None:
    _require_backends()
    task_table = _table({"entity": ["A", "C"]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "items",
        "table_column": "item_id",
    }

    expected = _sample(data, task_table, task_link, [-1, -1])
    actual = _sample(
        data.cuda(),
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
                {"account": [1, 2], "region": ["europe", "tokyo"]},
                {"account": Stype.id, "region": Stype.id},
            ),
            "orders": _table(
                {
                    "order_id": [10, 11],
                    "account": [1, 2],
                    "region": ["europe", "tokyo"],
                },
                {
                    "order_id": Stype.id,
                    "account": Stype.id,
                    "region": Stype.id,
                },
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_columns": ("account", "region"),
                "right_table": "users",
                "right_columns": ("account", "region"),
            }
        ],
    )
    task_table = _table(
        {"account": [2], "region": ["tokyo"]},
        {"account": Stype.id, "region": Stype.id},
    )
    task_link = {
        "task_columns": ("account", "region"),
        "table": "users",
        "table_columns": ("account", "region"),
    }

    expected = _sample(data, task_table, task_link, [-1])
    actual = _sample(
        data.cuda(),
        _cuda_table(task_table),
        task_link,
        [-1],
    )

    assert _canonical_output(actual) == _canonical_output(expected)


@onlyCUDA
def test_pyg_and_cugraph_match_temporal_last_samples(
    temporal_data: RelationalData,
) -> None:
    _require_backends()
    task_table = _table(
        {
            "entity": [0, 0],
            "cutoff": pd.to_datetime([2, 10], unit="s").tolist(),
        },
        {"entity": Stype.id, "cutoff": Stype.datetime},
    )
    task_link = {
        "task_column": "entity",
        "table": "roots",
        "table_column": "root_id",
    }
    time_columns = {"first": "time", "second": "time"}

    expected = _sample(
        temporal_data,
        task_table,
        task_link,
        [1, 1],
        time_columns=time_columns,
        task_time_column="cutoff",
    )
    actual = _sample(
        temporal_data.cuda(),
        _cuda_table(task_table),
        task_link,
        [1, 1],
        time_columns=time_columns,
        task_time_column="cutoff",
    )

    assert _canonical_output(actual) == _canonical_output(expected)


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
    data: RelationalData,
) -> None:
    _require_backends()
    task_table = _table({"entity": [0]}, {"entity": Stype.id})
    task_link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }

    cpu_all = _sample(data, task_table, task_link, [-1])
    cpu_sample = _sample(data, task_table, task_link, [1])
    gpu_data = data.cuda()
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
