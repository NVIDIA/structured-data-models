from datetime import datetime, timedelta
from typing import cast

import pytest
import torch

import sdm.processing as sp
from sdm import RelatedTables, Stype, TableTensor, Task
from sdm.callbacks import Callback
from sdm.models import KumoRelational
from sdm.models.kumo.relational.model import _KumoRelational
from sdm.models.kumo.relational.task import TaskGraph, _QueryGraphCache
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def make_query(
    start: int,
    rows: int,
    hops: int,
    device: torch.device,
) -> tuple[TableTensor, RelatedTables[TableTensor]]:
    ids = list(range(start, start + rows))
    times = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(rows)]
    x = TableTensor.from_columns(
        {"id": ids, "time": times, "feature": [float(i % 3) for i in ids]},
        stypes={"id": "id", "time": "datetime", "feature": "numerical"},
        device=device,
    )
    tables = {
        "users": TableTensor.from_columns(
            {"id": ids, "value": [float(i % 5) for i in ids]},
            stypes={"id": "id", "value": "numerical"},
            device=device,
        ),
        "orders": TableTensor.from_columns(
            {
                "id": ids,
                "user_id": ids if hops >= 1 else [-1] * rows,
                "value": [float(i % 4) for i in ids],
                "time": times,
            },
            stypes={
                "id": "id",
                "user_id": "id",
                "value": "numerical",
                "time": "datetime",
            },
            device=device,
        ),
        "lines": TableTensor.from_columns(
            {
                "order_id": ids if hops >= 2 else [-1] * rows,
                "value": [float(i % 7) for i in ids],
            },
            stypes={"order_id": "id", "value": "numerical"},
            device=device,
        ),
    }
    return x, RelatedTables(
        tables=tables,
        relationships=[
            {
                "left_table": "orders",
                "left_columns": "user_id",
                "right_table": "users",
                "right_columns": "id",
            },
            {
                "left_table": "lines",
                "left_columns": "order_id",
                "right_table": "orders",
                "right_columns": "id",
            },
        ],
        task_links=[
            {
                "task_columns": "id",
                "table": "users",
                "table_columns": "id",
            }
        ],
    )


def fit_model(
    hops: int | None,
    device: torch.device,
    transform_ids: bool = False,
    num_estimators: int | None = 3,
) -> KumoRelational:
    model = KumoRelational(task="regression", pretrained=False, device="meta")
    model.models[Task.regression] = _KumoRelational(
        num_classes=0,
        num_quantiles=999,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        device=device,
    ).eval()
    # Nonzero residual projections expose stale features and task assignments.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
    contexts = [make_query(100 * i, 5 + i, i, device) for i in range(3)]
    targets = [
        TableTensor(numerical=torch.randn(5 + i, 1, device=device))
        for i in range(3)
    ]
    related = contexts[0][1]
    recipe = model.default_recipe()
    if transform_ids:
        recipe = recipe.prepend_features(sp.StypeDispatch(id=sp.Identity()))
    model.fit(
        x=EnsembleTable.from_tables([x for x, _ in contexts], (0, 1, 2)),
        y=EnsembleTable.from_tables(targets, (0, 1, 2)),
        related_tables=RelatedTables(
            tables={
                name: EnsembleTable.from_tables(
                    [r.tables[name] for _, r in contexts], (0, 1, 2)
                )
                for name in related.tables
            },
            relationships=related.relationships,
            task_links=related.task_links,
        ),
        recipe=recipe,
        num_estimators=num_estimators,
        num_hops=hops,
    )
    return model


@withCUDA
@pytest.mark.parametrize("hops", [0, 1, 2, None])
@pytest.mark.parametrize("shared", [True, False])
def test_predict_graph_reuse(
    hops: int | None,
    shared: bool,
    device: torch.device,
) -> None:
    model = fit_model(hops, device)
    for rows, query_hops in ((2, 2), (4, 2), (2, 0)):
        x, related = make_query(1000, rows, query_hops, device)
        query = EnsembleTable(x, num_members=3)
        if shared:
            related_query = RelatedTables(
                tables={
                    name: EnsembleTable(table, num_members=3)
                    for name, table in related.tables.items()
                },
                relationships=related.relationships,
                task_links=related.task_links,
            )
        else:
            related_query = RelatedTables(
                tables={
                    name: EnsembleTable.from_tables(
                        [table, table[list(reversed(range(rows)))], table[:1]],
                        (0, 1, 2),
                    )
                    if name != "users"
                    else EnsembleTable(table, num_members=3)
                    for name, table in related.tables.items()
                },
                relationships=related.relationships,
                task_links=related.task_links,
            )
        # Callbacks use the unchanged per-member graph construction path.
        expected = model.predict(query, related_query, callbacks=[Callback()])
        actual = model.predict(query, related_query)
        torch.testing.assert_close(
            actual.numerical, expected.numerical, rtol=0, atol=0
        )
        if shared:
            raw = model.predict(x, related)
            torch.testing.assert_close(
                raw.numerical, expected.numerical, rtol=0, atol=0
            )


@withCUDA
def test_predict_batched_query_tables(device: torch.device) -> None:
    model = fit_model(2, device, num_estimators=None)
    queries = [make_query(1000 + 100 * i, 3, 2, device) for i in range(3)]
    x = cast(
        TableTensor,
        torch.stack([queries[0][0], queries[1][0][[2, 1, 0]], queries[2][0]]),
    )
    related = RelatedTables(
        tables={
            name: cast(
                TableTensor,
                torch.stack([related.tables[name] for _, related in queries]),
            )
            for name in queries[0][1].tables
        },
        relationships=queries[0][1].relationships,
        task_links=queries[0][1].task_links,
    )
    expected = model.predict(x, related, callbacks=[Callback()])
    actual = model.predict(x, related)
    torch.testing.assert_close(
        actual.numerical, expected.numerical, rtol=0, atol=0
    )


@withCUDA
def test_predict_id_processor(device: torch.device) -> None:
    model = fit_model(None, device, transform_ids=True)
    x, related = make_query(1000, 3, 2, device)
    x = EnsembleTable(x, num_members=3)
    related_query = RelatedTables(
        tables={
            name: EnsembleTable(table, num_members=3)
            for name, table in related.tables.items()
        },
        relationships=related.relationships,
        task_links=related.task_links,
    )
    expected = model.predict(x, related_query, callbacks=[Callback()])
    actual = model.predict(x, related_query)
    torch.testing.assert_close(
        actual.numerical, expected.numerical, rtol=0, atol=0
    )


@withCUDA
@pytest.mark.parametrize("readout", ["users", "orders"])
def test_graph_cache_metadata(readout: str, device: torch.device) -> None:
    x, related = make_query(1000, 3, 2, device)
    cache = _QueryGraphCache({tuple(related.relationships)})
    for tables in (
        related.tables,
        dict(reversed(list(related.tables.items()))),
        {**related.tables, "lines": related.tables["lines"][:1]},
    ):
        for relationships in (
            related.relationships,
            tuple(reversed(related.relationships)),
        ):
            current = RelatedTables(
                tables=tables,
                relationships=relationships,
                task_links=[
                    {
                        "task_columns": "id",
                        "table": readout,
                        "table_columns": "id",
                    }
                ],
            )
            for hops in (0, 1, 2):
                expected = TaskGraph.from_input(x, current, hops)
                actual = cache.from_input(x, current, hops)
                assert actual.num_hops == expected.num_hops
                assert actual.readout_table == expected.readout_table
                assert (
                    actual.graph.start_node_offsets
                    == expected.graph.start_node_offsets
                )
                for name in ("row", "col", "colptr", "edge_type"):
                    torch.testing.assert_close(
                        getattr(actual.graph, name),
                        getattr(expected.graph, name),
                    )
                torch.testing.assert_close(
                    actual.readout_index, expected.readout_index
                )
                for name in tables:
                    torch.testing.assert_close(
                        actual.task_row_by_table[name],
                        expected.task_row_by_table[name],
                    )


@withCUDA
def test_graph_reuse_requires_original_task_ids(device: torch.device) -> None:
    model = fit_model(2, device)
    x, related = make_query(1000, 3, 2, device)
    raw = TableTensor.from_arrow(
        x.to_arrow(),
        stypes={**x.stypes, "id": Stype.numerical},
        device=device,
    )
    assert model._predict_kwargs(raw, related) == {}


@withCUDA
@pytest.mark.parametrize("join_stype", [Stype.numerical, Stype.id])
def test_graph_cache_non_id_source(
    join_stype: Stype, device: torch.device
) -> None:
    x = TableTensor.from_columns(
        {"key": [0.0, 1.0]}, stypes={"key": join_stype}, device=device
    )
    cache = _QueryGraphCache(set())
    for order in ([0, 1], [1, 0]):
        related = RelatedTables(
            tables={"users": x[order]},
            relationships=[],
            task_links=[
                {
                    "task_columns": "key",
                    "table": "users",
                    "table_columns": "key",
                }
            ],
        )
        expected = TaskGraph.from_input(x, related, 0)
        actual = cache.from_input(x, related, 0)
        torch.testing.assert_close(
            actual.readout_index, expected.readout_index
        )
