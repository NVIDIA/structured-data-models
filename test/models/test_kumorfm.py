from typing import cast

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
from sdm.cache import Cache
from sdm.models import KumoRFM
from sdm.models.kumorfm import model as kumorfm_model
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.model import _KumoRFM, _remap_v2_1_checkpoint
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


def test_fit_predict_not_supported() -> None:
    model = KumoRFM(pretrained=False)
    caches = cast("list[Cache]", ["sentinel"])
    model._caches = caches

    with pytest.raises(NotImplementedError, match="does not support 'fit"):
        model.fit(torch.randn(4, 2), torch.randn(4, 1))
    assert model._caches is caches

    with pytest.raises(NotImplementedError, match="does not support 'fit"):
        model.predict(torch.randn(4, 2))


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

    graph = HomogeneousGraph.from_related_tables(related_tables)
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


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(
    relational_data: RelationalData,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model = KumoRFM(False, device)
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

    out = model(
        x_context=x,
        y_context=y,
        x_query=x,
        related_context_tables=related_tables,
        related_query_tables=related_tables,
        num_hops=2,
    )

    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.is_inference(out)


def test_too_many_classes() -> None:
    num_users = 11
    users = TableTensor(
        columns={Stype.numerical: ("age",), Stype.id: ("user_id",)},
        numerical=torch.randn(num_users, 1),
        id=ColumnarTensor((torch.arange(num_users),)),
    )
    orders = TableTensor(
        columns={Stype.numerical: ("amount",), Stype.id: ("user_id",)},
        numerical=torch.randn(2 * num_users, 1),
        id=ColumnarTensor((torch.arange(2 * num_users) % num_users,)),
    )
    related_tables = RelatedTables(
        tables={"users": users, "orders": orders},
        relationships=[
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
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
        id=ColumnarTensor((torch.arange(num_users),)),
    )
    y = TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.arange(num_users).view(-1, 1),
            categories=(torch.arange(num_users),),
        ),
    )
    model = KumoRFM(pretrained=False)

    with pytest.raises(NotImplementedError, match="at most 10 classes"):
        model(
            x_context=x,
            y_context=y,
            x_query=x,
            related_context_tables=related_tables,
            related_query_tables=related_tables,
            num_hops=2,
        )


def test_forward_honors_generator(relational_data: RelationalData) -> None:
    model = KumoRFM(pretrained=False)

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
        id=ColumnarTensor((torch.arange(4),)),
    )
    y = TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [0], [1]]),
            categories=(torch.tensor([False, True]),),
        ),
    )

    def _call(seed: int) -> torch.Tensor:
        out = model(
            x_context=x,
            y_context=y,
            x_query=x,
            related_context_tables=related_tables,
            related_query_tables=related_tables,
            num_hops=2,
            generator=torch.Generator().manual_seed(seed),
        )
        return out.numerical

    assert _call(0).equal(_call(0))
    assert not _call(0).equal(_call(1))


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
