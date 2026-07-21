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
from sdm.models.kumorfm.model import _KumoRFM
from sdm.testing import withCUDA


def test_load_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads: list[dict[str, object]] = []
    loads: list[tuple[str, object, bool]] = []
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

    monkeypatch.setattr(kumorfm_model, "download_checkpoint", download)
    monkeypatch.setattr(kumorfm_model.torch, "load", load)
    monkeypatch.setattr(_KumoRFM, "load_state_dict", load_state_dict)

    model = KumoRFM(
        repo_id="org/private-model",
        revision="v1",
        cache_dir="/cache",
        local_files_only=True,
    )

    assert not model.training
    assert downloads == [
        {
            "repo_id": "org/private-model",
            "filename": "classifier.ckpt",
            "revision": "v1",
            "cache_dir": "/cache",
            "local_files_only": True,
        },
        {
            "repo_id": "org/private-model",
            "filename": "regressor.ckpt",
            "revision": "v1",
            "cache_dir": "/cache",
            "local_files_only": True,
        },
    ]
    assert [path for path, _, _ in loads] == [
        "/classifier.ckpt",
        "/regressor.ckpt",
    ]
    assert all(weights_only for _, _, weights_only in loads)
    assert state_dicts == [
        ({"/classifier.ckpt": torch.tensor(1)}, True),
        ({"/regressor.ckpt": torch.tensor(1)}, True),
    ]


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
