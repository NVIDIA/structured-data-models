# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pandas as pd
import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    RelatedTables,
    RelationalData,
    Stype,
    TableTensor,
)
from sdm.cache import Cache, Int8KVCacheEntry, KVCacheEntry
from sdm.models import KumoRelational
from sdm.models.kumo.relational.graph import HomogeneousGraph
from sdm.models.kumo.relational.invariant_gnn import InvariantGNN
from sdm.models.kumo.relational.model import (
    _KumoRelational,
)
from sdm.testing import withCUDA


@withCUDA
def test_invariant_gnn(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    related_tables = RelatedTables(
        tables={
            "users": relational_data.tables["users"],
            "orders": relational_data.tables["orders"],
        },
        relationships=relational_data.relationships[:1],
        task_links=[],
    )

    graph = HomogeneousGraph.from_tables(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
    )
    assert graph.col.equal(
        torch.tensor([0, 0, 1, 3, 3, 3, 4, 5, 6, 7, 8, 9], device=device)
    )
    assert graph.colptr.equal(
        torch.tensor([0, 2, 3, 3, 6, 7, 8, 9, 10, 11, 12], device=device)
    )
    # Make `row` deterministic within local neighborhoods:
    row, perm = graph.row.sort()
    row = row[graph.col[perm].argsort(stable=True)]
    assert row.equal(
        torch.tensor([4, 5, 6, 7, 8, 9, 0, 0, 1, 3, 3, 3], device=device)
    )
    assert graph.edge_type.equal(
        torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], device=device)
    )
    assert graph.num_edge_types == 2
    assert graph.start_node_offsets == {"users": 0, "orders": 4}
    assert graph.end_node_offsets == {"users": 4, "orders": 10}

    model = InvariantGNN(channels=8, device=device)
    out = model(
        x=torch.randn(10, 8, device=device),
        graph=graph,
        readout_table="users",
        readout_index=torch.arange(4, device=device),
        num_hops=2,
    )
    assert out.size() == (4, 8)
    assert out.device == device
    assert not out.isnan().any()


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_invariant_gnn_autocast(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # Float32 parameters under autocast, as in the RelBench example.
    related_tables = RelatedTables(
        tables={
            "users": relational_data.tables["users"],
            "orders": relational_data.tables["orders"],
        },
        relationships=relational_data.relationships[:1],
        task_links=[],
    )
    graph = HomogeneousGraph.from_tables(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
    )
    model = InvariantGNN(channels=8, device=device)

    with torch.inference_mode(), torch.autocast(device.type, dtype):
        out = model(
            x=torch.randn(10, 8, device=device),
            graph=graph,
            readout_table="users",
            readout_index=torch.arange(4, device=device),
            num_hops=2,
        )

    assert out.size() == (4, 8)
    assert out.device == device
    assert not out.isnan().any()


@withCUDA
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
)
def test_invariant_gnn_cache(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    related_tables = RelatedTables(
        tables={
            "users": relational_data.tables["users"],
            "orders": relational_data.tables["orders"],
        },
        relationships=relational_data.relationships[:1],
        task_links=[],
    )
    graph = HomogeneousGraph.from_tables(
        tables=related_tables.tables,
        relationships=related_tables.relationships,
    )
    model = InvariantGNN(channels=8, device=device, dtype=dtype)
    x = torch.randn(10, 8, device=device, dtype=dtype)
    readout_index = torch.arange(4, device=device)

    with torch.inference_mode():
        expected = model(
            x=x,
            graph=graph,
            readout_table="users",
            readout_index=readout_index,
            num_hops=2,
            generator=torch.Generator(device=device).manual_seed(0),
        )
        cache = Cache()
        recorded = model(
            x=x,
            graph=graph,
            readout_table="users",
            readout_index=readout_index,
            num_hops=2,
            cache=cache,
            generator=torch.Generator(device=device).manual_seed(0),
        )
        replayed = model(
            x=x,
            graph=graph,
            readout_table="users",
            readout_index=readout_index,
            num_hops=2,
            cache=cache.freeze(),
        )

    torch.testing.assert_close(recorded, expected)
    torch.testing.assert_close(replayed, expected)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model = KumoRelational(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "KumoRelational()"
    else:
        assert repr(model) == "KumoRelational(device=cuda:0)"

    related_tables = RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )

    x = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "timestamp": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", None, "2024-01-06"]
                ),
            }
        ),
        stypes={"user_id": "id", "timestamp": "datetime"},
        device=device,
    )

    if dtype.is_floating_point:
        y = TableTensor(
            numerical=torch.randn(4, 1, device=device),
        )
    else:
        y = TableTensor(
            categorical=CategoricalTensor(
                code=torch.randint(0, 2, size=(4, 1), device=device),
                categories=(torch.tensor([False, True], device=device),),
            ),
        )

    torch.manual_seed(1)
    out = model(
        x_context=x,
        y_context=y,
        x_query=x,
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        num_hops=2,
    )

    assert out.size(-2) == 4
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.is_inference(out)
    if dtype.is_floating_point:
        assert (out.numerical.diff(dim=-1) >= 0).all()

    assert (
        model(
            x_context=x,
            y_context=y,
            x_query=x,
            related_context_tables=related_tables,
            related_query_tables=related_tables,
            num_hops=0,
        ).size()
        == out.size()
    )

    torch.manual_seed(1)
    model.fit(x, y, related_tables)
    assert model.predict(
        x=x,
        related_tables=RelatedTables(
            tables=dict(reversed(list(related_tables.tables.items()))),
            relationships=related_tables.relationships[::-1],
            task_links=related_tables.task_links[::-1],
        ),
    ).allclose(out, atol=1e-4, rtol=1e-4)
    model.clear()


@withCUDA
def test_many_classes_forward_and_cache(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    num_classes = 3
    ids = torch.arange(4, device=device)
    classes = torch.arange(num_classes, device=device)
    task = TableTensor(
        columns={Stype.id: ("user_id",)},
        id=ColumnarTensor((ids,)),
    )
    target = TableTensor(
        categorical=CategoricalTensor(
            code=ids.remainder(num_classes).to(torch.int32).unsqueeze(-1),
            categories=(classes,),
        ),
    )
    related_tables = RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )
    model = _KumoRelational(
        num_classes=2,
        num_quantiles=0,
        channels=4,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=2,
        group_size=2,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
        device=device,
    )

    expected = model(
        x_context=task,
        y_context=target,
        x_query=task[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        num_hops=0,
    )
    assert expected.size() == (2, num_classes)
    probabilities = expected.div(0.9).exp()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        expected.new_ones(2),
    )

    cache = Cache(classes=classes)
    recorded = model(
        x_context=task,
        y_context=target,
        x_query=None,
        related_context_tables=related_tables,
        related_query_tables=None,
        cache=cache,
        num_hops=0,
    )
    assert recorded.size() == (0, num_classes)

    predicted = model(
        x_context=None,
        y_context=None,
        x_query=task[:2],
        related_context_tables=None,
        related_query_tables=related_tables,
        cache=cache.freeze(),
    )
    torch.testing.assert_close(predicted, expected)


def test_int8_kv_cache(relational_data: RelationalData) -> None:
    model = KumoRelational(pretrained=False)

    related_tables = RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )
    x = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "timestamp": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", None, "2024-01-06"]
                ),
            }
        ),
        stypes={"user_id": "id", "timestamp": "datetime"},
    )
    y = TableTensor(numerical=torch.randn(4, 1))

    model.fit(x, y, related_tables)
    assert model._cache is not None
    default_size = model._cache.size()
    expected = model.predict(x, related_tables)

    # Key/value entries recorded in caches the model creates internally must
    # honor the configured storage dtype too.
    model.fit(x, y, related_tables, kv_cache_dtype=torch.int8)
    assert model._cache is not None
    assert model._cache.size() < default_size

    def key_value_entries(cache: Cache) -> Iterator[object]:
        for value in cache.values():
            if isinstance(value, KVCacheEntry | Int8KVCacheEntry):
                yield value
            elif isinstance(value, Cache):
                yield from key_value_entries(value)

    entries = list(key_value_entries(model._cache))
    assert len(entries) > 0
    assert all(isinstance(entry, Int8KVCacheEntry) for entry in entries)

    out = model.predict(x, related_tables)
    assert out.size() == expected.size()
    assert out.numerical.isfinite().all()
    model.clear()
