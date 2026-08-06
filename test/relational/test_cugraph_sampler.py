from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest
import torch

from sdm import (
    ColumnarTensor,
    RelationalData,
    Stype,
    TableTensor,
    TemporalSamplingConfig,
)
from sdm.relational import CuGraphRelationalSampler
from sdm.relational.sampler import EXAMPLE_ID
from sdm.testing import onlyCUDA


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


def _id_table(
    column: str,
    values: list[int],
    dtype: torch.dtype,
) -> TableTensor:
    return TableTensor(
        columns={Stype.id: (column,)},
        id=ColumnarTensor(
            columns=(torch.tensor(values, dtype=dtype, device="cuda"),)
        ),
    )


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


def _rows(table: TableTensor, *columns: str) -> list[tuple[Any, ...]]:
    values = (
        cast(TableTensor, table.cpu()).to_arrow().select(columns).to_pydict()
    )
    return sorted(zip(*(values[column] for column in columns)))


@onlyCUDA
def test_cuda_data_uses_cugraph_sampler() -> None:
    _require_rapids()

    sampler = _non_temporal_data().sampler()

    assert isinstance(sampler, CuGraphRelationalSampler)


@onlyCUDA
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


@onlyCUDA
def test_cugraph_sampler_advances_seeded_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_rapids()
    data = _non_temporal_data()
    first = CuGraphRelationalSampler(data=data, random_state=123)
    second = CuGraphRelationalSampler(data=data, random_state=123)
    task_table = _table({"entity": [0]}, {"entity": Stype.id})
    states: list[int] = []
    sample = first._pylibcugraph.heterogeneous_uniform_neighbor_sample

    def capture(*args: Any, **kwargs: Any) -> Any:
        states.append(kwargs["random_state"])
        return sample(*args, **kwargs)

    monkeypatch.setattr(
        first._pylibcugraph,
        "heterogeneous_uniform_neighbor_sample",
        capture,
    )

    def sample_orders(
        sampler: CuGraphRelationalSampler,
    ) -> list[tuple[Any, ...]]:
        output = sampler(
            task_table=task_table,
            task_link={
                "task_column": "entity",
                "table": "users",
                "table_column": "user_id",
            },
            num_neighbors=[1],
        )
        return _rows(output.related_tables.tables["orders"], "order_id")

    first_sequence = [sample_orders(first), sample_orders(first)]
    second_sequence = [sample_orders(second), sample_orders(second)]

    assert states[0] != states[1]
    assert states[:2] == states[2:]
    assert first_sequence == second_sequence


@onlyCUDA
def test_cugraph_sampler_retains_seed_when_relationship_is_empty() -> None:
    _require_rapids()
    data = _non_temporal_data()
    data = RelationalData(
        tables={**data.tables, "orders": data.tables["orders"][:0]},
        relationships=data.relationships,
    )
    task_table = _table({"entity": [2]}, {"entity": Stype.id})

    output = data.sampler()(
        task_table=task_table,
        task_link={
            "task_column": "entity",
            "table": "users",
            "table_column": "user_id",
        },
        num_neighbors=[-1],
    )

    assert _rows(
        output.related_tables.tables["users"], EXAMPLE_ID, "user_id"
    ) == [(0, 2)]


@onlyCUDA
def test_cugraph_sampler_resolves_composite_string_seed() -> None:
    _require_rapids()
    data = RelationalData(
        tables={
            "users": _table(
                {"account": [1, 2], "region": ["é", "東京"]},
                {"account": Stype.id, "region": Stype.id},
            ),
            "orders": _table(
                {
                    "order_id": [10, 11],
                    "account": [1, 2],
                    "region": ["é", "東京"],
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
        {"account": [2], "region": ["東京"]},
        {"account": Stype.id, "region": Stype.id},
    )

    output = data.sampler()(
        task_table=task_table,
        task_link={
            "task_columns": ("account", "region"),
            "table": "users",
            "table_columns": ("account", "region"),
        },
        num_neighbors=[-1],
    )

    assert _rows(
        output.related_tables.tables["orders"], EXAMPLE_ID, "order_id"
    ) == [(0, 11)]


@onlyCUDA
def test_cugraph_sampler_resolves_numeric_seed_without_cudf_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_rapids()
    data = RelationalData(
        tables={
            "users": _table(
                {"user_id": [30, 10, 20]},
                {"user_id": Stype.id},
            )
        },
        relationships=[],
    )
    sampler = data.sampler()
    task_table = _table({"entity": [20, 30]}, {"entity": Stype.id})

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("numeric seed lookup used the cuDF join fallback")

    monkeypatch.setattr("sdm.relational.cugraph_sampler.join_index", fail)

    output = sampler(
        task_table=task_table,
        task_link={
            "task_column": "entity",
            "table": "users",
            "table_column": "user_id",
        },
        num_neighbors=[0],
    )

    assert _rows(
        output.related_tables.tables["users"], EXAMPLE_ID, "user_id"
    ) == [(0, 20), (1, 30)]


@onlyCUDA
def test_cugraph_sampler_does_not_match_null_seed_to_zero() -> None:
    _require_rapids()
    data = RelationalData(
        tables={
            "users": _table(
                {"user_id": [0, 1]},
                {"user_id": Stype.id},
            )
        },
        relationships=[],
    )
    task_table = TableTensor.from_arrow(
        table=pa.table({"entity": pa.array([None], type=pa.int64())}),
        stypes={"entity": Stype.id},
        device="cuda",
    )

    with pytest.raises(ValueError, match="match exactly one row"):
        data.sampler()(
            task_table=task_table,
            task_link={
                "task_column": "entity",
                "table": "users",
                "table_column": "user_id",
            },
            num_neighbors=[0],
        )


@onlyCUDA
def test_cugraph_sampler_invalidates_mutated_numeric_seed_lookup() -> None:
    _require_rapids()
    data = RelationalData(
        tables={
            "users": _table(
                {"user_id": [30, 10, 20]},
                {"user_id": Stype.id},
            )
        },
        relationships=[],
    )
    sampler = data.sampler()
    link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }

    sampler(
        task_table=_table({"entity": [30]}, {"entity": Stype.id}),
        task_link=link,
        num_neighbors=[0],
    )
    data.tables["users"].id.unbind(-1)[0][0] = 40

    with pytest.raises(ValueError, match="match exactly one row"):
        sampler(
            task_table=_table({"entity": [30]}, {"entity": Stype.id}),
            task_link=link,
            num_neighbors=[0],
        )

    output = sampler(
        task_table=_table({"entity": [40]}, {"entity": Stype.id}),
        task_link=link,
        num_neighbors=[0],
    )
    assert _rows(
        output.related_tables.tables["users"], EXAMPLE_ID, "user_id"
    ) == [(0, 40)]


@onlyCUDA
def test_cugraph_sampler_invalidates_retyped_numeric_seed_lookup() -> None:
    _require_rapids()
    tables = {"users": _id_table("user_id", [-1, 1], torch.int8)}
    data = RelationalData(
        tables=tables,
        relationships=[],
    )
    sampler = data.sampler()
    link = {
        "task_column": "entity",
        "table": "users",
        "table_column": "user_id",
    }
    sampler(
        task_table=_id_table("entity", [-1], torch.int8),
        task_link=link,
        num_neighbors=[0],
    )

    source = data.tables["users"].id.unbind(-1)[0]
    tables["users"] = TableTensor(
        columns={Stype.id: ("user_id",)},
        id=ColumnarTensor(columns=(source.view(torch.uint8),)),
    )

    output = sampler(
        task_table=_id_table("entity", [255], torch.uint8),
        task_link=link,
        num_neighbors=[0],
    )
    assert _rows(
        output.related_tables.tables["users"], EXAMPLE_ID, "user_id"
    ) == [(0, 255)]


@onlyCUDA
def test_cugraph_temporal_sampler_uses_bounded_uniform_fanout(
    temporal_data: RelationalData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_rapids()
    sampler = CuGraphRelationalSampler(
        data=temporal_data.cuda(),
        temporal=TemporalSamplingConfig(
            time_columns={"first": "time", "second": "time"},
            strategy="uniform",
        ),
    )
    sample = (
        sampler._pylibcugraph.heterogeneous_uniform_temporal_neighbor_sample
    )
    fanouts: list[list[int]] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        fanouts.append(args[7].tolist())
        return sample(*args, **kwargs)

    monkeypatch.setattr(
        sampler._pylibcugraph,
        "heterogeneous_uniform_temporal_neighbor_sample",
        capture,
    )
    sampler(
        task_table=_table(
            {
                "entity": [0],
                "cutoff": pd.to_datetime([10], unit="s").tolist(),
            },
            {"entity": Stype.id, "cutoff": Stype.datetime},
        ),
        task_link={
            "task_column": "entity",
            "table": "roots",
            "table_column": "root_id",
        },
        num_neighbors=[1, 2],
        task_time_column="cutoff",
    )

    assert fanouts == [[1, 0, 0, 0], [0, 2, 2, 0]]


@onlyCUDA
def test_cugraph_sampler_uses_original_cutoff(
    temporal_data: RelationalData,
) -> None:
    _require_rapids()
    data = temporal_data.cuda()
    task_table = _table(
        {
            "entity": [0, 0],
            "cutoff": pd.to_datetime([2, 10], unit="s").tolist(),
        },
        {"entity": Stype.id, "cutoff": Stype.datetime},
    )

    output = data.sampler(
        temporal=TemporalSamplingConfig(
            time_columns={"first": "time", "second": "time"},
            strategy="uniform",
        )
    )(
        task_table=task_table,
        task_link={
            "task_column": "entity",
            "table": "roots",
            "table_column": "root_id",
        },
        num_neighbors=[-1, -1],
        task_time_column="cutoff",
    )

    assert _rows(
        output.related_tables.tables["first"], EXAMPLE_ID, "first_id"
    ) == [(0, 11), (0, 12), (1, 10), (1, 11), (1, 12)]
    # The second-hop time (8) is newer than the first-hop time (3), but is
    # valid because every hop uses the task cutoff (10).
    assert _rows(
        output.related_tables.tables["second"], EXAMPLE_ID, "second_id"
    ) == [(1, 20), (1, 21)]


@onlyCUDA
def test_cugraph_temporal_sampler_rejects_last_strategy(
    temporal_data: RelationalData,
) -> None:
    _require_rapids()

    with pytest.raises(NotImplementedError, match="strategy 'last'"):
        temporal_data.cuda().sampler(
            temporal=TemporalSamplingConfig(
                time_columns={"first": "time", "second": "time"},
                strategy="last",
            )
        )


@onlyCUDA
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


@onlyCUDA
def test_cugraph_sampler_does_not_export_rows_to_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_rapids()
    import cudf  # noqa: PLC0415

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
