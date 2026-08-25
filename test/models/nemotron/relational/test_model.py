import warnings
from typing import cast

import pandas as pd
import pytest
import torch
from torch import Tensor

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    RelatedTables,
    RelationalData,
    Stype,
    TableTensor,
)
from sdm.cache import Cache
from sdm.models import NemotronRelational
from sdm.models.nemotron.relational import invariant_gnn as gnn_module
from sdm.models.nemotron.relational import model as model_module
from sdm.models.nemotron.relational.graph import HomogeneousGraph
from sdm.models.nemotron.relational.invariant_gnn import InvariantGNN
from sdm.models.nemotron.relational.model import (
    _NemotronRelational,
    _remap_v2_1_checkpoint,
)
from sdm.models.nemotron.relational.training_free_gnn import (
    _fit_table_encoder,
    _fit_transform_training_free_gnn,
    _message,
    _transform_training_free_gnn,
)
from sdm.tensor.mixin import DeviceMixin
from sdm.testing import withCUDA


def test_load_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[dict[str, object]] = []
    loads: list[tuple[str, object, bool]] = []
    remaps: list[tuple[object, bool]] = []
    state_dicts: list[tuple[object, bool]] = []

    def download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return f"/{kwargs['filename']}"

    def load(
        path: str,
        *,
        map_location: object,
        weights_only: bool,
    ) -> dict[str, object]:
        loads.append((path, map_location, weights_only))
        return {"state_dict": {path: torch.tensor(1)}}

    def load_state_dict(
        self: _NemotronRelational,
        state_dict: object,
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        state_dicts.append((state_dict, strict))

    def remap(
        state_dict: object,
        *,
        is_classifier: bool,
    ) -> dict[str, torch.Tensor]:
        remaps.append((state_dict, is_classifier))
        return {str(is_classifier): torch.tensor(1)}

    monkeypatch.setattr(model_module, "download_checkpoint", download)
    monkeypatch.setattr(model_module.torch, "load", load)
    monkeypatch.setattr(model_module, "_remap_v2_1_checkpoint", remap)
    monkeypatch.setattr(
        _NemotronRelational,
        "load_state_dict",
        load_state_dict,
    )

    model = NemotronRelational()

    assert not model.training
    assert model.reg_model.row_embedding.norm.bias is not None
    assert model.reg_model.icl_block.norm.bias is not None
    assert downloads == [
        {
            "repo_id": "nvidia/kumorfm",
            "filename": "cls-model.pt",
            "revision": "v2.1.0",
        },
        {
            "repo_id": "nvidia/kumorfm",
            "filename": "reg-model.pt",
            "revision": "v2.1.0",
        },
    ]
    assert [path for path, _, _ in loads] == [
        "/cls-model.pt",
        "/reg-model.pt",
    ]
    assert all(weights_only for _, _, weights_only in loads)
    assert remaps == [
        ({"/cls-model.pt": torch.tensor(1)}, True),
        ({"/reg-model.pt": torch.tensor(1)}, False),
    ]
    assert state_dicts == [
        ({"True": torch.tensor(1)}, True),
        ({"False": torch.tensor(1)}, True),
    ]


@pytest.mark.parametrize(
    ("is_classifier", "source", "target"),
    [
        (True, "row_embedding.y_reg_lin.weight", None),
        (True, "icl_block.reg_head.weight", None),
        (False, "row_embedding.y_cls_lin.weight", None),
        (False, "icl_block.cls_head.weight", None),
        (
            False,
            "row_embedding.y_reg_lin.weight",
            "row_embedding.y_lin.weight",
        ),
        (False, "icl_block.y_reg_lin.bias", "icl_block.y_lin.bias"),
        (
            False,
            "icl_block.reg_head.weight",
            "icl_block.head.2.weight",
        ),
    ],
)
def test_remap_v2_1_variant_keys(
    is_classifier: bool,
    source: str,
    target: str | None,
) -> None:
    value = torch.tensor(1)
    expected = {} if target is None else {target: value}

    actual = _remap_v2_1_checkpoint(
        {source: value},
        is_classifier=is_classifier,
    )

    assert actual == expected


def test_remap_v2_1_checkpoint_shapes() -> None:
    inducing_points = torch.arange(6).reshape(2, 1, 3)
    readout_token = torch.arange(6).reshape(1, 2, 3)

    actual = _remap_v2_1_checkpoint(
        {
            "q": torch.tensor([0.02, 0.98]),
            "row_embedding.inducing_vectors.2": inducing_points,
            "row_embedding.readout_token": readout_token,
        },
        is_classifier=True,
    )

    assert actual.keys() == {
        "row_embedding.col_layers.2.inducing_points",
        "row_embedding.readout_token",
    }
    torch.testing.assert_close(
        actual["row_embedding.col_layers.2.inducing_points"],
        inducing_points.squeeze(1),
    )
    torch.testing.assert_close(
        actual["row_embedding.readout_token"],
        readout_token.squeeze(0),
    )


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
    model = NemotronRelational(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "NemotronRelational()"
    else:
        assert repr(model) == "NemotronRelational(device=cuda:0)"

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
    model = _NemotronRelational(
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


def _training_free_gnn_input(
    relational_data: RelationalData,
) -> tuple[TableTensor, RelatedTables]:
    x = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "segment": ["a", "b", "a", "c"],
                "timestamp": pd.to_datetime(
                    ["2024-01-03", "2024-01-04", None, "2024-01-06"]
                ),
            }
        ),
        stypes={
            "user_id": "id",
            "segment": "categorical",
            "timestamp": "datetime",
        },
        device=relational_data.device,
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
    return x, related_tables


def test_training_free_gnn_relation_operators() -> None:
    channels = 64
    x = torch.zeros(6, channels)
    x[0, 0] = 1
    x[1, 0] = 3
    x[2, 0] = 5
    weight = torch.zeros(channels, 5 * channels)
    weight[0, torch.arange(5) * channels] = 1
    bias = torch.ones(channels)
    out = _message(
        x=x,
        edge_type=0,
        row=torch.tensor([0, 1, 2]),
        colptr=torch.tensor([0, 2, 3, 3]),
        state=Cache(weight=weight, bias=bias),
    )
    torch.testing.assert_close(out[:, 0], torch.tensor([12.0, 21.0, 1.0]))
    torch.testing.assert_close(out[:, 1:], torch.ones(3, channels - 1))

    x[3] = 2
    x[4] = 5
    out = _message(
        x=x,
        edge_type=1,
        row=torch.tensor([3, 4]),
        colptr=torch.tensor([0, 1, 2, 2]),
        state=Cache(weight=torch.eye(channels), bias=bias),
    )
    torch.testing.assert_close(
        out,
        torch.tensor([[3.0] * channels, [6.0] * channels, [1.0] * channels]),
    )

    id_only = TableTensor.from_columns(
        {"id": [0, 1]},
        stypes={"id": "id"},
    )
    _, id_embedding = _fit_table_encoder(
        id_only,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(id_embedding, torch.ones(2, channels))

    empty_datetime = TableTensor(
        columns={Stype.datetime: ("time",)},
        datetime=torch.empty((0, 1), dtype=torch.int64),
    )
    _, empty_embedding = _fit_table_encoder(
        empty_datetime,
        generator=torch.Generator().manual_seed(42),
    )
    assert empty_embedding.size() == (0, channels)

    numerical = TableTensor.from_columns(
        {"a": [0.0, 1.0], "b": [1.0, 0.0]},
        stypes={"a": "numerical", "b": "numerical"},
    )
    encoder_state, expected = _fit_table_encoder(
        numerical,
        generator=torch.Generator().manual_seed(42),
    )
    weight = encoder_state["continuous_weight"]
    assert isinstance(weight, Tensor)
    assert not weight[:, 0].equal(weight[:, 1])
    reordered = TableTensor(
        columns={Stype.numerical: ("b", "a")},
        numerical=numerical.numerical[:, [1, 0]],
    )
    _, actual = _fit_table_encoder(
        reordered,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(actual, expected)


@withCUDA
def test_training_free_gnn_determinism_and_schema_order(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    assert relational_data.device == device
    x, related_tables = _training_free_gnn_input(relational_data)
    schemas = {
        name: table.schema for name, table in related_tables.tables.items()
    }
    x_schema = x.schema
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    expected, expected_state = _fit_transform_training_free_gnn(
        x=x,
        related_tables=related_tables,
        num_hops=2,
    )
    assert torch.random.get_rng_state().equal(rng_state)
    if cuda_rng_state is not None:
        assert torch.cuda.get_rng_state(device).equal(cuda_rng_state)

    reversed_tables = dict(reversed(list(related_tables.tables.items())))
    reversed_related_tables = RelatedTables(
        tables=reversed_tables,
        relationships=related_tables.relationships[::-1],
        task_links=related_tables.task_links,
    )
    actual, actual_state = _fit_transform_training_free_gnn(
        x=x,
        related_tables=reversed_related_tables,
        num_hops=2,
    )

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual_state.size() == expected_state.size()
    assert expected.size() == (4, 64)
    assert expected.numerical.isfinite().all()
    assert x.schema == x_schema
    assert {
        name: table.schema for name, table in related_tables.tables.items()
    } == schemas


def test_training_free_gnn_unknown_category_and_zero_hop(
    relational_data: RelationalData,
) -> None:
    x, related_tables = _training_free_gnn_input(relational_data)
    _, state = _fit_transform_training_free_gnn(
        x=x,
        related_tables=related_tables,
        num_hops=0,
    )
    task_state = cast(Cache, state["task_state"])
    task_categories = cast(tuple[Tensor, ...], task_state["categories"])
    categories = tuple(category.clone() for category in task_categories)
    query = TableTensor.from_pandas(
        df=pd.DataFrame(
            {
                "user_id": [0, 1],
                "segment": ["unseen", None],
                "timestamp": pd.to_datetime(["2024-02-01", None]),
            }
        ),
        stypes={
            "user_id": "id",
            "segment": "categorical",
            "timestamp": "datetime",
        },
        device=relational_data.device,
    )
    out = _transform_training_free_gnn(
        x=query,
        related_tables=related_tables,
        state=state,
    )

    assert out.size() == (2, 64)
    assert out.numerical.isfinite().all()
    after_categories = cast(tuple[Tensor, ...], task_state["categories"])
    for before, after in zip(categories, after_categories, strict=True):
        assert after.equal(before)

    subset = related_tables.select_tables(["users"])
    subset_out = _transform_training_free_gnn(
        x=query,
        related_tables=subset,
        state=state.to(query.device),
    )
    assert subset_out.size() == (2, 64)
    assert subset_out.numerical.isfinite().all()


@withCUDA
def test_training_free_gnn_lifecycle_runs_once_before_estimators(
    relational_data: RelationalData,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, related_tables = _training_free_gnn_input(relational_data)
    y = TableTensor(numerical=torch.arange(4.0, device=device).unsqueeze(-1))
    model = NemotronRelational(
        pretrained=False,
        device=device,
        training_free_gnn_features=True,
    )
    state_keys = model.state_dict().keys()
    calls: list[Tensor] = []
    fit_feature_calls = 0
    original_fit_features = model_module._fit_transform_training_free_gnn

    def fit_features(
        *,
        x: TableTensor,
        related_tables: RelatedTables,
        num_hops: int | None,
    ) -> tuple[TableTensor, Cache]:
        nonlocal fit_feature_calls
        fit_feature_calls += 1
        return original_fit_features(
            x=x,
            related_tables=related_tables,
            num_hops=num_hops,
        )

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
        if x_context is not None:
            calls.append(x_context.numerical[:, -64:].clone())
        if x_query is None:
            assert x_context is not None
            return TableTensor.from_tensor(x_context.numerical[:0, :1])
        return TableTensor.from_tensor(
            x_query.numerical[:, -64:].sum(dim=-1, keepdim=True)
        )

    monkeypatch.setattr(model, "_forward", fake_forward)
    monkeypatch.setattr(
        model_module,
        "_fit_transform_training_free_gnn",
        fit_features,
    )
    recipe = sp.Recipe(
        features=sp.Identity(),
        target=sp.Standardize(),
        output=sp.ReduceEstimators(method="mean"),
    )
    warn_always = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = model(
                x_context=x,
                y_context=y,
                x_query=x[:2],
                related_context_tables=related_tables,
                related_query_tables=related_tables,
                recipe=recipe,
                num_estimators=3,
                num_hops=2,
            )
    finally:
        torch.set_warn_always(warn_always)
    assert not any(
        "unsupported feature stypes" in str(warning.message)
        for warning in caught
    )

    assert len(calls) == 3
    assert fit_feature_calls == 1
    for actual in calls[1:]:
        torch.testing.assert_close(actual, calls[0])

    context_features = calls[0]
    calls.clear()
    model(
        x_context=x,
        y_context=TableTensor(numerical=-y.numerical),
        x_query=x[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        recipe=recipe,
        num_estimators=3,
        num_hops=2,
    )
    torch.testing.assert_close(calls[0], context_features)
    assert fit_feature_calls == 2

    calls.clear()
    model.fit(
        x=x,
        y=y,
        related_tables=related_tables,
        recipe=recipe,
        num_estimators=3,
        num_hops=2,
    )
    assert fit_feature_calls == 3
    assert model._cache is not None
    cached_feature_state = model._cache["task_feature_state"]
    assert isinstance(cached_feature_state, Cache)
    assert cached_feature_state.device == x.device
    replayed_feature_states: list[DeviceMixin | None] = []
    original_transform_features = model._transform_task_features

    def transform_features(
        *,
        x: TableTensor,
        related_tables: RelatedTables | None,
        state: DeviceMixin | None,
        **kwargs: object,
    ) -> TableTensor | None:
        replayed_feature_states.append(state)
        return original_transform_features(
            x=x,
            related_tables=related_tables,
            state=state,
            **kwargs,
        )

    monkeypatch.setattr(model, "_transform_task_features", transform_features)
    actual = model.predict(x[:2], related_tables)
    repeated = model.predict(x[:2], related_tables)
    torch.testing.assert_close(actual.numerical, expected.numerical)
    torch.testing.assert_close(repeated.numerical, expected.numerical)
    assert len(replayed_feature_states) == 2
    assert all(
        state is cached_feature_state for state in replayed_feature_states
    )
    assert model.state_dict().keys() == state_keys


def test_training_free_gnn_disabled(
    relational_data: RelationalData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, related_tables = _training_free_gnn_input(relational_data)
    model = NemotronRelational(False, relational_data.device)
    del model.training_free_gnn_features

    features, state = model._fit_task_features(
        x,
        related_tables,
        num_hops=2,
    )

    assert features is None
    assert state is None
    assert all("training_free" not in key for key in model.state_dict())

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
        assert x_query is not None
        return TableTensor.from_tensor(
            x_query.numerical.sum(dim=-1, keepdim=True)
        )

    monkeypatch.setattr(model, "_forward", fake_forward)
    y = TableTensor(numerical=torch.arange(4.0).unsqueeze(-1))
    expected = model(
        x_context=x,
        y_context=y,
        x_query=x[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        generator=torch.Generator().manual_seed(0),
        num_hops=2,
    )

    def no_fit_features(
        x: TableTensor,
        related_tables: RelatedTables | None,
        **kwargs: object,
    ) -> tuple[TableTensor | None, DeviceMixin | None]:
        return None, None

    def no_transform_features(
        x: TableTensor,
        related_tables: RelatedTables | None,
        state: DeviceMixin | None,
        **kwargs: object,
    ) -> TableTensor | None:
        return None

    monkeypatch.setattr(model, "_fit_task_features", no_fit_features)
    monkeypatch.setattr(
        model,
        "_transform_task_features",
        no_transform_features,
    )
    actual = model(
        x_context=x,
        y_context=y,
        x_query=x[:2],
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        generator=torch.Generator().manual_seed(0),
        num_hops=2,
    )
    assert actual.numerical.equal(expected.numerical)
