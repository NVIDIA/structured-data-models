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
from sdm.explain import (
    CaptumIntegratedGradients,
    FeatureAttribution,
    GradientSensitivity,
    InputSite,
    IntegratedGradientsDiagnostics,
    OutputIndex,
)
from sdm.models import KumoRFM
from sdm.models.kumorfm import model as kumorfm_model
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.model import _KumoRFM, _remap_v2_1_checkpoint
from sdm.testing import withCUDA
from test.models._explain import FittedEndpointMethod, cache_tensors


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
        self: _KumoRFM,
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

    monkeypatch.setattr(kumorfm_model, "download_checkpoint", download)
    monkeypatch.setattr(kumorfm_model.torch, "load", load)
    monkeypatch.setattr(kumorfm_model, "_remap_v2_1_checkpoint", remap)
    monkeypatch.setattr(_KumoRFM, "load_state_dict", load_state_dict)

    model = KumoRFM()

    assert not model.training
    assert model.reg_model.row_embedding.norm.bias is not None
    assert model.reg_model.icl_block.norm.bias is not None
    assert downloads == [
        {
            "repo_id": "nvidia/kumorfm-2",
            "filename": "cls-model.pt",
            "revision": "v2.1.0",
        },
        {
            "repo_id": "nvidia/kumorfm-2",
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
        (False, "icl_block.reg_head.weight", "head.2.weight"),
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
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model = KumoRFM(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "KumoRFM()"
    else:
        assert repr(model) == "KumoRFM(device=cuda:0)"

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

    x = TableTensor(
        columns={"id": ("user_id",)},
        id=ColumnarTensor((torch.arange(4, device=device),)),
    )

    if dtype.is_floating_point:
        y = TableTensor(
            columns={"numerical": ("target",)},
            numerical=torch.randn(4, 1, device=device),
        )
    else:
        y = TableTensor(
            columns={"categorical": ("target",)},
            categorical=CategoricalTensor(
                data=torch.randint(0, 2, size=(4, 1), device=device),
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


def test_default_recipe_preserves_ids() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: ("entity_id",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        id=ColumnarTensor((torch.tensor([10, 11]),)),
    )

    transformed = KumoRFM.default_recipe().features.fit_transform(table)

    assert transformed.columns[Stype.id] == ("entity_id",)
    assert transformed.id is table.id


def test_explain_fitted_replays_kumorfm_related_cache(
    relational_data: RelationalData,
) -> None:
    model = KumoRFM(pretrained=False).eval()
    related = RelatedTables(
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
    x = TableTensor(
        columns={Stype.id: ("user_id",)},
        id=ColumnarTensor((torch.arange(4),)),
    )
    y = TableTensor.from_tensor(
        torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
        columns=("target",),
    )
    torch.manual_seed(1)
    model.fit(x, y, related, num_hops=2)
    prediction = model.predict(x, related)
    caches = model._caches
    assert caches is not None
    tensors = cache_tensors(caches)
    snapshot = tuple(tensor.clone() for tensor in tensors)
    cache_sizes = tuple(cache.size() for cache in caches)

    explanation = model.explain_fitted(
        FittedEndpointMethod(),
        x,
        related,
        target=OutputIndex(row=0, column="q500"),
    )

    assert explanation.prediction.allclose(
        prediction,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert model._caches is caches
    assert all(cache.is_replaying and cache.is_cpu for cache in caches)
    assert tuple(cache.size() for cache in caches) == cache_sizes
    assert tuple(id(tensor) for tensor in cache_tensors(caches)) == tuple(
        id(tensor) for tensor in tensors
    )
    for actual, expected in zip(tensors, snapshot):
        assert actual.equal(expected)


def _small_kumorfm_core(
    *,
    num_classes: int,
    num_quantiles: int,
) -> _KumoRFM:
    core = _KumoRFM(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=2,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
        device="cpu",
    )
    with torch.no_grad():
        for path in (
            "row_embedding.row_layers.0.attn.out_lin",
            "icl_block.layers.0.attn.out_lin",
        ):
            projection = core.get_submodule(path)
            assert isinstance(projection, torch.nn.Linear)
            projection.weight.fill_(0.1)
    return core


def _kumorfm_explanation_case(
    relational_data: RelationalData,
) -> tuple[
    KumoRFM,
    tuple[
        TableTensor,
        TableTensor,
        TableTensor,
        RelatedTables,
        RelatedTables,
    ],
    RelatedTables,
]:
    model = KumoRFM(pretrained=False, device="meta")
    model.cls_model = _small_kumorfm_core(
        num_classes=10,
        num_quantiles=0,
    )
    model.reg_model = _small_kumorfm_core(
        num_classes=0,
        num_quantiles=999,
    )
    model.eval()
    related = RelatedTables(
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
    x_context = TableTensor(
        columns={
            Stype.numerical: ("ignored",),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor([[-2.0], [-1.0], [0.0], [1.0]]),
        id=ColumnarTensor((torch.arange(4),)),
    )
    x_query = TableTensor(
        columns={
            Stype.numerical: ("ignored",),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor([[2.0]]),
        id=ColumnarTensor((torch.tensor([0]),)),
    )
    y_context = TableTensor.from_tensor(
        torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
        columns=("target",),
    )
    args = (x_context, y_context, x_query, related, related)
    return model, args, related


def test_gradient_sensitivity_uses_kumorfm_related_inputs(
    relational_data: RelationalData,
) -> None:
    model, args, related = _kumorfm_explanation_case(relational_data)
    expected = model(
        *args,
        generator=torch.Generator().manual_seed(3),
        num_hops=2,
    )

    explanation = model.explain_full_context(
        GradientSensitivity(
            magnitude=True,
            normalization="global_max_abs",
        ),
        *args,
        generator=torch.Generator().manual_seed(3),
        num_hops=2,
        target=OutputIndex(row=0, column="q500"),
    )

    assert explanation.prediction.schema == expected.schema
    torch.testing.assert_close(
        explanation.prediction.numerical,
        expected.numerical,
    )
    assert explanation.target.column == 499
    assert all(parameter.grad is None for parameter in model.parameters())
    attributions: dict[InputSite, FeatureAttribution] = {
        attribution.site: attribution
        for attribution in explanation.attributions
    }
    assert InputSite(split="context", table="users") in attributions
    assert InputSite(split="query", table="users") in attributions

    maxima: list[torch.Tensor] = []
    for site, attribution in attributions.items():
        assert attribution.input_space == "processed"
        assert attribution.score_kind == "gradient"
        assert not attribution.signed
        assert attribution.normalization == "global_max_abs"
        if site.table is not None:
            original = related.tables[site.table]
            assert (
                attribution.values.columns[Stype.id]
                == (original.columns[Stype.id])
            )
            for actual, expected_id in zip(
                attribution.values.id.unbind(-1),
                original.id.unbind(-1),
            ):
                assert actual.equal(expected_id)
        if attribution.values.numerical.numel() > 0:
            maxima.append(attribution.values.numerical.abs().amax())

    assert torch.stack(maxima).amax().item() == pytest.approx(1.0)


def test_captum_integrated_gradients_uses_kumorfm_execution(
    relational_data: RelationalData,
) -> None:
    pytest.importorskip("captum")
    model, args, _ = _kumorfm_explanation_case(relational_data)
    query_site = InputSite(split="query")
    expected = model(
        *args,
        generator=torch.Generator().manual_seed(3),
        num_hops=2,
    )

    explanation = model.explain_full_context(
        CaptumIntegratedGradients(
            baselines={query_site: torch.zeros(1, 1)},
            n_steps=4,
        ),
        *args,
        generator=torch.Generator().manual_seed(3),
        num_hops=2,
        target=OutputIndex(row=0, column="q500"),
    )

    assert explanation.prediction.allclose(expected)
    assert explanation.target.column == 499
    assert all(parameter.grad is None for parameter in model.parameters())
    assert len(explanation.attributions) == 1
    attribution = explanation.attributions[0]
    assert attribution.site == query_site
    assert attribution.score_kind == "integrated_gradients"
    assert attribution.input_space == "processed"
    assert attribution.values.size() == (1, 2)
    assert torch.isfinite(attribution.values.numerical).all()
    assert len(explanation.diagnostics) == 1
    diagnostics = explanation.diagnostics[0]
    assert isinstance(diagnostics, IntegratedGradientsDiagnostics)
    assert diagnostics.varied_sites == (query_site,)
    assert diagnostics.n_steps == 4
    assert torch.isfinite(diagnostics.convergence_delta).all()
