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
from sdm.models import KumoRFM
from sdm.models.kumorfm import model as kumorfm_model
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.model import (
    _KumoRFM,
    _propagate_targets,
    _remap_v2_1_checkpoint,
)
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
    assert graph.colptr.equal(
        torch.tensor([0, 2, 3, 3, 6, 7, 8, 9, 10, 11, 12], device=device)
    )
    col = torch.repeat_interleave(
        torch.arange(graph.colptr.numel() - 1, device=graph.colptr.device),
        graph.colptr.diff(),
    )
    # Make `row` deterministic within local neighborhoods:
    row, perm = graph.row.sort()
    row = row[col[perm].argsort(stable=True)]
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
        edge_type_emb=model.get_edge_type_emb(graph.num_edge_types),
        readout_table="users",
        num_hops=2,
    )
    assert out.size() == (4, 8)
    assert out.device == device
    assert not out.isnan().any()

    isolated_graph = HomogeneousGraph.from_related_tables(
        related_tables.select_tables(tables=["users"])
    )
    isolated_x = torch.randn(4, 8, device=device)
    isolated_out = model(
        x=isolated_x,
        graph=isolated_graph,
        edge_type_emb=model.get_edge_type_emb(isolated_graph.num_edge_types),
        readout_table="users",
        num_hops=1,
    )
    expected = model.out_norm(
        model.out_lin(
            torch.nn.functional.gelu(model.norm(model.skip_lin(isolated_x)))
        )
    )
    torch.testing.assert_close(isolated_out, expected)

    full_related_tables = RelatedTables(
        tables=relational_data.tables,
        relationships=relational_data.relationships,
        task_links=[],
    )
    subset = full_related_tables.select_tables(tables=["orders", "items"])
    subset_graph = HomogeneousGraph.from_related_tables(
        subset,
        relationship_order=full_related_tables.relationships,
    )
    assert subset_graph.edge_type.unique().equal(
        torch.tensor([2, 3], device=device)
    )
    assert subset_graph.num_edge_types == 4


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

    x_context = TableTensor(
        columns={
            "numerical": ("task_feature",),
            "id": ("user_id",),
        },
        numerical=torch.tensor(
            [[0.5], [1.5], [2.5], [3.5]],
            device=device,
        ),
        id=ColumnarTensor((torch.tensor([3, 1, 2, 0], device=device),)),
    )
    x_query = TableTensor(
        columns={
            "numerical": ("task_feature",),
            "id": ("user_id",),
        },
        numerical=torch.tensor([[2.5], [3.5]], device=device),
        id=ColumnarTensor((torch.tensor([2, 0], device=device),)),
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
                data=torch.tensor([[0], [1], [0], [1]], device=device),
                categories=(torch.tensor([False, True], device=device),),
            ),
        )

    torch.manual_seed(1)
    out = model(
        x_context=x_context,
        y_context=y,
        x_query=x_query,
        related_context_tables=related_tables,
        related_query_tables=related_tables.select_tables(tables=["users"]),
        num_hops=2,
    )

    assert out.size(-2) == 2
    assert out.dtype == x_query.dtype
    assert out.device == x_query.device
    assert torch.is_inference(out)

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


@pytest.mark.parametrize(
    ("y", "num_classes", "expected"),
    [
        (
            torch.tensor([1, 0]),
            2,
            torch.tensor([1, 0, 0, 0]),
        ),
        (
            torch.tensor([2.0, 6.0]),
            0,
            torch.tensor([2.0, 4.0, 6.0, 6.0]),
        ),
    ],
)
def test_propagate_targets(
    y: torch.Tensor,
    num_classes: int,
    expected: torch.Tensor,
) -> None:
    graph = HomogeneousGraph(
        row=torch.tensor([1, 0, 2, 1, 3, 2]),
        colptr=torch.tensor([0, 1, 3, 5, 6]),
        edge_type=torch.zeros(6, dtype=torch.long),
        num_edge_types=1,
        start_node_offsets={"table": 0},
        end_node_offsets={"table": 4},
    )

    out = _propagate_targets(
        y=y,
        root_index=torch.tensor([0, 2]),
        graph=graph,
        num_hops=1,
        num_classes=num_classes,
        dtype=torch.float32,
    )

    assert torch.equal(out, expected)

    disconnected = HomogeneousGraph(
        row=graph.row[:4],
        colptr=torch.tensor([0, 1, 3, 4, 4]),
        edge_type=graph.edge_type[:4],
        num_edge_types=graph.num_edge_types,
        start_node_offsets=graph.start_node_offsets,
        end_node_offsets=graph.end_node_offsets,
    )
    with pytest.raises(ValueError, match="did not propagate"):
        _propagate_targets(
            y=y,
            root_index=torch.tensor([0, 2]),
            graph=disconnected,
            num_hops=1,
            num_classes=num_classes,
            dtype=torch.float32,
        )


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
