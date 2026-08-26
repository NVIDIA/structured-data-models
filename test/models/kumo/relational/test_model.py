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
from sdm.cache import Cache, KVCacheEntry
from sdm.models import KumoRelational
from sdm.models.kumo.relational import invariant_gnn as gnn_module
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
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
)
def test_invariant_gnn_destination_chunks(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
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
    model = InvariantGNN(channels=8, device=device, dtype=dtype).eval()
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
        monkeypatch.setattr(
            gnn_module,
            "_automatic_aggregation_work_byte_limit",
            lambda _x, _graph: 1024,
        )
        actual = model(
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

    torch.testing.assert_close(actual, expected)
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
    ).allclose(out)
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
    ).eval()

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

    cache = Cache(classes=classes, kv_cache_offload="layer")
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

    entries: list[KVCacheEntry] = []
    for value in cache.values():
        if isinstance(value, KVCacheEntry):
            entries.append(value)
        elif isinstance(value, Cache):
            entries.extend(
                nested
                for nested in value.values()
                if isinstance(nested, KVCacheEntry)
            )
    assert entries
    if device.type == "cuda":
        assert all(entry.key.is_cpu for entry in entries)
        assert all(entry.value.is_cpu for entry in entries)
        assert all(
            entry.key.numel() == 0 or entry.key.is_pinned()
            for entry in entries
        )
        assert all(
            entry.value.numel() == 0 or entry.value.is_pinned()
            for entry in entries
        )

    predicted = model(
        x_context=None,
        y_context=None,
        x_query=task[:2],
        related_context_tables=None,
        related_query_tables=related_tables,
        cache=cache.freeze().to(device, non_blocking=True),
    )
    torch.testing.assert_close(predicted, expected)
