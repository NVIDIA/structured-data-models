from typing import Any, cast

import pandas as pd
import pytest
import torch
from sdm import RelationalData, Stype, TableTensor
from sdm.relational import CuGraphRelationalSampler
from sdm.relational.sampler import EXAMPLE_ID


def _require_rapids() -> None:
    if not torch.cuda.is_available():
        cast(Any, pytest.skip)("CUDA is not available")
    pytest.importorskip("cudf")
    pytest.importorskip("pylibcugraph")


def _table(
    data: dict[str, list[Any]],
    stypes: dict[str, Stype],
) -> TableTensor:
    table = TableTensor.from_pandas(df=pd.DataFrame(data), stypes=stypes)
    return cast(TableTensor, table.cuda())


def _non_temporal_data() -> RelationalData:
    return RelationalData(
        tables={
            "users": _table(
                {"user_id": [0, 1, 2], "score": [0.1, 0.2, 0.3]},
                {"user_id": Stype.id, "score": Stype.numerical},
            ),
            "orders": _table(
                {
                    "order_id": [10, 11, 12],
                    "user_id": [0, 0, 1],
                    "amount": [1.0, 2.0, 3.0],
                },
                {
                    "order_id": Stype.id,
                    "user_id": Stype.id,
                    "amount": Stype.numerical,
                },
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
    )


def _temporal_data() -> RelationalData:
    return RelationalData(
        tables={
            "roots": _table(
                {"root_id": [0]},
                {"root_id": Stype.id},
            ),
            "first": _table(
                {
                    "first_id": [10, 11, 12],
                    "root_id": [0, 0, 0],
                    "time": pd.to_datetime([3, 1, 2], unit="s").tolist(),
                },
                {
                    "first_id": Stype.id,
                    "root_id": Stype.id,
                    "time": Stype.datetime,
                },
            ),
            "second": _table(
                {
                    "second_id": [20, 21],
                    "first_id": [10, 12],
                    "time": pd.to_datetime([8, 9], unit="s").tolist(),
                },
                {
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


def _rows(table: TableTensor, *columns: str) -> list[tuple[Any, ...]]:
    values = (
        cast(TableTensor, table.cpu())
        .to_arrow()
        .select(columns)
        .to_pydict()
    )
    return sorted(zip(*(values[column] for column in columns)))


def test_cuda_data_uses_cugraph_sampler() -> None:
    _require_rapids()

    sampler = _non_temporal_data().sampler()

    assert isinstance(sampler, CuGraphRelationalSampler)


def test_cugraph_sampler_is_disjoint_and_retains_isolated_seeds() -> None:
    _require_rapids()
    data = _non_temporal_data()
    task_table = _table(
        {"entity": [0, 2, 0]},
        {"entity": Stype.id},
    )

    output = data.sampler()(
        task_table=task_table,
        task_link={
            "task_column": "entity",
            "table": "users",
            "table_column": "user_id",
        },
        num_neighbors=[-1],
    )

    assert output.device.type == "cuda"
    assert _rows(
        output.related_tables.tables["users"], EXAMPLE_ID, "user_id"
    ) == [(0, 0), (1, 2), (2, 0)]
    assert _rows(
        output.related_tables.tables["orders"], EXAMPLE_ID, "order_id"
    ) == [(0, 10), (0, 11), (2, 10), (2, 11)]


def test_cugraph_sampler_uses_original_cutoff_and_latest_neighbors() -> None:
    _require_rapids()
    data = _temporal_data()
    task_table = _table(
        {
            "entity": [0, 0],
            "cutoff": pd.to_datetime([2, 10], unit="s").tolist(),
        },
        {"entity": Stype.id, "cutoff": Stype.datetime},
    )

    output = data.sampler(
        time_columns={"first": "time", "second": "time"}
    )(
        task_table=task_table,
        task_link={
            "task_column": "entity",
            "table": "roots",
            "table_column": "root_id",
        },
        num_neighbors=[1, 1],
        task_time_column="cutoff",
    )

    assert _rows(
        output.related_tables.tables["first"], EXAMPLE_ID, "first_id"
    ) == [(0, 12), (1, 10)]
    # The second-hop time (8) is newer than the first-hop time (3), but is
    # valid because every hop uses the task cutoff (10).
    assert _rows(
        output.related_tables.tables["second"], EXAMPLE_ID, "second_id"
    ) == [(1, 20)]


@pytest.mark.parametrize("entity", [99, 0], ids=["missing", "duplicate"])
def test_cugraph_sampler_requires_one_seed_match(entity: int) -> None:
    _require_rapids()
    data = _non_temporal_data()
    if entity == 0:
        users = torch.cat([data.tables["users"], data.tables["users"][:1]])
        data = RelationalData(
            tables={**data.tables, "users": cast(TableTensor, users)},
            relationships=data.relationships,
        )
    task_table = _table({"entity": [entity]}, {"entity": Stype.id})

    with pytest.raises(ValueError, match="match exactly one row"):
        data.sampler()(
            task_table=task_table,
            task_link={
                "task_column": "entity",
                "table": "users",
                "table_column": "user_id",
            },
            num_neighbors=[-1],
        )


def test_cugraph_sampler_does_not_export_rows_to_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_rapids()
    import cudf

    data = _non_temporal_data()
    sampler = data.sampler()
    task_table = _table({"entity": [0]}, {"entity": Stype.id})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("cuGraph sampling exported row data to host")

    with monkeypatch.context() as host_guard:
        host_guard.setattr(TableTensor, "to_arrow", fail)
        host_guard.setattr(cudf.DataFrame, "to_arrow", fail)
        host_guard.setattr(cudf.DataFrame, "to_pandas", fail)
        host_guard.setattr(cudf.Series, "to_numpy", fail)
        host_guard.setattr(torch.Tensor, "cpu", fail)
        host_guard.setattr(torch.Tensor, "numpy", fail)
        host_guard.setattr(torch.Tensor, "tolist", fail)

        output = sampler(
            task_table=task_table,
            task_link={
                "task_column": "entity",
                "table": "users",
                "table_column": "user_id",
            },
            num_neighbors=[-1],
        )
        torch.cuda.synchronize()

    assert output.device.type == "cuda"
