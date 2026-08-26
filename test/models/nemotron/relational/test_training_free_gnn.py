import pickle
from typing import cast

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm import NaT, RelatedTables, RelationalData, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import NemotronRelational
from sdm.models.nemotron.relational import model as model_module
from sdm.models.nemotron.relational.task import TaskGraph
from sdm.models.nemotron.relational.training_free_gnn import (
    _CHANNELS,
    _embed,
    _fit_projection,
    _fit_state,
    _fit_transform_training_free_gnn,
    _message,
    _project,
    _transform_training_free_gnn,
)
from sdm.testing import withCUDA


def _input(
    relational_data: RelationalData,
) -> tuple[TableTensor, RelatedTables]:
    x = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "balance": [0.0, 1.0, 4.0, 10.0],
                "segment": ["a", "b", "a", "c"],
                "timestamp": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", None, "2024-01-06"]
                ),
                "reviewed_at": pd.to_datetime(
                    ["2024-01-08", None, "2024-01-02", "2024-01-09"]
                ),
            }
        ),
        stypes={
            "user_id": "id",
            "balance": "numerical",
            "segment": "categorical",
            "timestamp": "datetime",
            "reviewed_at": "datetime",
        },
        device=relational_data.device,
    )
    tables = dict(relational_data.tables)
    orders = tables["orders"]
    order_columns = dict(orders.columns)
    order_columns[Stype.datetime] = (
        *order_columns[Stype.datetime],
        "settled_at",
    )
    settled_at = torch.where(
        orders.datetime == NaT,
        orders.datetime,
        orders.datetime + 7 * 24 * 60 * 60 * 1_000_000,
    )
    tables["orders"] = TableTensor(
        columns=order_columns,
        numerical=orders.numerical,
        categorical=orders.categorical,
        datetime=torch.cat((orders.datetime, settled_at), dim=-1),
        text=orders.text,
        id=orders.id,
    )
    return x, RelatedTables(
        tables=tables,
        relationships=relational_data.relationships,
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )


def _reverse_features(table: TableTensor) -> TableTensor:
    columns = dict(table.columns)
    columns[Stype.numerical] = tuple(reversed(columns[Stype.numerical]))
    columns[Stype.datetime] = tuple(reversed(columns[Stype.datetime]))
    return TableTensor(
        columns=columns,
        numerical=table.numerical.flip(-1),
        categorical=table.categorical,
        datetime=table.datetime.flip(-1),
        text=table.text,
        id=table.id,
    )


def test_directional_operators_and_empty_projection() -> None:
    x = torch.zeros(6, _CHANNELS)
    x[0, 0] = 1
    x[1, 0] = 3
    x[2, 0] = 5
    weight = torch.zeros(_CHANNELS, 5 * _CHANNELS)
    weight[0, torch.arange(5) * _CHANNELS] = 1
    bias = torch.ones(_CHANNELS)

    out = _message(
        x=x,
        edge_type=0,
        row=torch.tensor([0, 1, 2]),
        colptr=torch.tensor([0, 2, 3, 3]),
        state=Cache(weight=weight, bias=bias),
    )
    torch.testing.assert_close(out[:, 0], torch.tensor([12.0, 21.0, 1.0]))
    torch.testing.assert_close(out[:, 1:], torch.ones(3, _CHANNELS - 1))

    x[3] = 2
    x[4] = 5
    out = _message(
        x=x,
        edge_type=1,
        row=torch.tensor([3, 4, 3]),
        colptr=torch.tensor([0, 2, 3, 3]),
        state=Cache(weight=torch.eye(_CHANNELS), bias=bias),
    )
    torch.testing.assert_close(
        out,
        torch.tensor(
            [
                [6.0] * _CHANNELS,
                [3.0] * _CHANNELS,
                [1.0] * _CHANNELS,
            ]
        ),
    )

    table = TableTensor(numerical=torch.empty(3, 0))
    state = _fit_projection(
        table,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(
        _project(table, state),
        torch.ones(3, _CHANNELS),
    )


def test_disjoint_component_message_parity() -> None:
    generator = torch.Generator().manual_seed(7)
    context_x = torch.randn(5, _CHANNELS, generator=generator)
    query_x = torch.randn(4, _CHANNELS, generator=generator)
    context_row = torch.tensor([0, 1, 2, 3])
    context_colptr = torch.tensor([0, 2, 3, 4])
    query_row = torch.tensor([0, 1, 2])
    query_colptr = torch.tensor([0, 1, 3])
    state = Cache(
        weight=torch.randn(
            _CHANNELS,
            5 * _CHANNELS,
            generator=generator,
        ),
        bias=torch.randn(_CHANNELS, generator=generator),
    )

    context = _message(
        x=context_x,
        edge_type=0,
        row=context_row,
        colptr=context_colptr,
        state=state,
    )
    query = _message(
        x=query_x,
        edge_type=0,
        row=query_row,
        colptr=query_colptr,
        state=state,
    )

    # Explicit block-diagonal CSR union; identifiers are never re-joined.
    joint_x = torch.cat((context_x, query_x))
    joint_row = torch.cat((context_row, query_row + context_x.size(0)))
    joint_colptr = torch.cat(
        (
            context_colptr,
            query_colptr[1:] + context_row.numel(),
        )
    )
    joint = _message(
        x=joint_x,
        edge_type=0,
        row=joint_row,
        colptr=joint_colptr,
        state=state,
    )
    expected = torch.cat((context, query))
    torch.testing.assert_close(joint, expected)
    torch.testing.assert_close(
        F.layer_norm(joint, (_CHANNELS,)),
        torch.cat(
            (
                F.layer_norm(context, (_CHANNELS,)),
                F.layer_norm(query, (_CHANNELS,)),
            )
        ),
    )


@withCUDA
def test_determinism_canonicalization_and_readout(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    x, related_tables = _input(relational_data)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )

    expected, state = _fit_transform_training_free_gnn(
        x=x,
        related_tables=related_tables,
        num_hops=2,
    )
    assert torch.random.get_rng_state().equal(cpu_rng)
    if cuda_rng is not None:
        assert torch.cuda.get_rng_state(device).equal(cuda_rng)
    assert expected.dtype == torch.float32
    assert expected.numerical.isfinite().all()
    assert expected.numerical.abs().max() <= 15

    reversed_tables = RelatedTables(
        tables=dict(reversed(list(related_tables.tables.items()))),
        relationships=related_tables.relationships[::-1],
        task_links=related_tables.task_links,
    )
    actual, actual_state = _fit_transform_training_free_gnn(
        x=_reverse_features(x),
        related_tables=RelatedTables(
            tables={
                name: _reverse_features(table)
                for name, table in reversed_tables.tables.items()
            },
            relationships=reversed_tables.relationships,
            task_links=reversed_tables.task_links,
        ),
        num_hops=2,
    )
    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual_state.size() == state.size()

    reordered = _transform_training_free_gnn(
        x=_reverse_features(x),
        related_tables=RelatedTables(
            tables={
                name: _reverse_features(table)
                for name, table in related_tables.tables.items()
            },
            relationships=related_tables.relationships,
            task_links=related_tables.task_links,
        ),
        state=state,
    )
    torch.testing.assert_close(reordered.numerical, expected.numerical)

    canonical = RelatedTables(
        tables={
            name: related_tables.tables[name]
            for name in sorted(related_tables.tables)
        },
        relationships=sorted(
            related_tables.relationships,
            key=lambda relationship: (
                relationship.left_table,
                relationship.left_columns,
                relationship.right_table,
                relationship.right_columns,
            ),
        ),
        task_links=related_tables.task_links,
    )
    task_graph = TaskGraph.from_input(
        x=x,
        related_tables=canonical,
        num_hops=0,
    )
    raw_state = _fit_state(x=x, task_graph=task_graph)
    raw = _embed(x=x, task_graph=task_graph, state=raw_state)
    table_state = cast(Cache, raw_state["table_projections"])
    entity = _project(
        canonical.tables[task_graph.readout_table],
        cast(Cache, table_state[task_graph.readout_table]),
    )[task_graph.readout_index]
    task = _project(x, cast(Cache, raw_state["task_projection"]))
    torch.testing.assert_close(raw, entity + task)


@withCUDA
def test_shared_feature_lifecycle(
    relational_data: RelationalData,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, related_tables = _input(relational_data)
    y = TableTensor(
        numerical=torch.tensor(
            [[0.0], [1.0], [3.0], [8.0]],
            device=device,
        )
    )
    model = NemotronRelational(
        pretrained=False,
        device=device,
        training_free_gnn_features=True,
    )
    state_keys = tuple(model.state_dict())
    calls: list[tuple[Tensor | None, Tensor | None]] = []
    feature_calls = {"fit": 0, "transform": 0}
    original_fit = model_module._fit_transform_training_free_gnn
    original_transform = model_module._transform_training_free_gnn

    def fit_feature(**kwargs: object) -> tuple[TableTensor, Cache]:
        feature_calls["fit"] += 1
        table = cast(TableTensor, kwargs["x"])
        tables = cast(RelatedTables, kwargs["related_tables"])
        assert Stype.categorical not in table.active_stypes
        assert all(
            Stype.categorical not in related.active_stypes
            for related in tables.tables.values()
        )
        return original_fit(**kwargs)  # type: ignore[arg-type]

    def transform_feature(**kwargs: object) -> TableTensor:
        feature_calls["transform"] += 1
        return original_transform(**kwargs)  # type: ignore[arg-type]

    def fake_forward(
        *,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: object,
    ) -> TableTensor:
        del y_context, related_context_tables, related_query_tables
        del cache, generator, kwargs
        calls.append(
            (
                x_context.numerical.clone() if x_context is not None else None,
                x_query.numerical.clone() if x_query is not None else None,
            )
        )
        if x_query is None:
            assert x_context is not None
            return TableTensor.from_tensor(x_context.numerical[:0, :1])
        return TableTensor.from_tensor(
            x_query.numerical[:, -_CHANNELS:].mean(dim=-1, keepdim=True)
        )

    monkeypatch.setattr(
        model_module,
        "_fit_transform_training_free_gnn",
        fit_feature,
    )
    monkeypatch.setattr(
        model_module,
        "_transform_training_free_gnn",
        transform_feature,
    )
    monkeypatch.setattr(model, "_forward", fake_forward)

    cpu_rng = torch.random.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    expected = model(
        x_context=x,
        y_context=y,
        x_query=x[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        num_estimators=8,
        generator=torch.Generator(device=device).manual_seed(0),
        num_hops=2,
    )
    assert feature_calls == {"fit": 1, "transform": 1}
    assert len(calls) == 8
    direct_calls = calls.copy()
    context_0, query_0 = cast(tuple[Tensor, Tensor], direct_calls[0])
    context_1, query_1 = cast(tuple[Tensor, Tensor], direct_calls[1])
    assert not context_0[:, :-_CHANNELS].equal(context_1[:, :-_CHANNELS])
    torch.testing.assert_close(
        context_0[:, -_CHANNELS:],
        context_1[:, -_CHANNELS:],
    )
    torch.testing.assert_close(
        query_0[:, -_CHANNELS:],
        query_1[:, -_CHANNELS:],
    )
    for context, query in direct_calls[2:]:
        assert context is not None
        assert query is not None
        torch.testing.assert_close(
            context[:, -_CHANNELS:],
            context_0[:, -_CHANNELS:],
        )
        torch.testing.assert_close(
            query[:, -_CHANNELS:],
            query_0[:, -_CHANNELS:],
        )

    calls.clear()
    model.fit(
        x=x,
        y=y,
        related_tables=related_tables,
        num_estimators=8,
        generator=torch.Generator(device=device).manual_seed(0),
        num_hops=2,
    )
    assert feature_calls == {"fit": 2, "transform": 1}
    assert model._cache is not None
    feature_state = model._cache["task_feature_state"]
    assert isinstance(feature_state, Cache)
    assert feature_state.device == device
    assert all(
        "task_feature_state" not in cast(Cache, model._cache[i])
        for i in range(8)
    )
    fit_context_0 = cast(Tensor, calls[0][0])
    torch.testing.assert_close(
        fit_context_0[:, -_CHANNELS:],
        context_0[:, -_CHANNELS:],
    )

    calls.clear()
    actual = model.predict(x[:2], related_tables)
    repeated = model.predict(x[:2], related_tables)
    torch.testing.assert_close(actual.numerical, expected.numerical)
    torch.testing.assert_close(repeated.numerical, expected.numerical)
    assert feature_calls == {"fit": 2, "transform": 3}
    for context, query in calls:
        assert context is None
        assert query is not None
        torch.testing.assert_close(
            query[:, -_CHANNELS:],
            query_0[:, -_CHANNELS:],
        )

    assert tuple(model.state_dict()) == state_keys
    assert torch.random.get_rng_state().equal(cpu_rng)
    if cuda_rng is not None:
        assert torch.cuda.get_rng_state(device).equal(cuda_rng)


def test_cached_state_pickle_and_legacy_disabled(
    relational_data: RelationalData,
) -> None:
    x, related_tables = _input(relational_data)
    expected, state = _fit_transform_training_free_gnn(
        x=x,
        related_tables=related_tables,
        num_hops=1,
    )
    restored = pickle.loads(pickle.dumps(state))
    actual = _transform_training_free_gnn(
        x=x,
        related_tables=related_tables,
        state=restored,
    )
    torch.testing.assert_close(actual.numerical, expected.numerical)

    model = NemotronRelational(pretrained=False)
    assert model._fit_task_features(x, related_tables)[0] is None
    del model.training_free_gnn_features
    assert model._fit_task_features(x, related_tables)[0] is None
    assert all("training_free" not in key for key in model.state_dict())


@withCUDA
def test_feature_enabled_model_forward_and_cache(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    x, related_tables = _input(relational_data)
    y = TableTensor(
        numerical=torch.tensor(
            [[0.0], [1.0], [3.0], [8.0]],
            device=device,
        )
    )
    model = NemotronRelational(
        pretrained=False,
        device=device,
        training_free_gnn_features=True,
    )

    expected = model(
        x_context=x,
        y_context=y,
        x_query=x[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        generator=torch.Generator(device=device).manual_seed(3),
        num_hops=2,
    )
    model.fit(
        x=x,
        y=y,
        related_tables=related_tables,
        generator=torch.Generator(device=device).manual_seed(3),
        num_hops=2,
    )
    actual = model.predict(x[:2], related_tables)

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual.size() == expected.size()
    assert actual.device == device
    assert model._cache is not None
    assert isinstance(model._cache["task_feature_state"], Cache)
