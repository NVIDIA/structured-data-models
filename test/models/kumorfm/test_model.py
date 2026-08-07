import copy
from datetime import UTC, datetime
from typing import Literal

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
from sdm.cache import Cache
from sdm.models import KumoRFM
from sdm.models.kumorfm import model as kumorfm_model
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.model import _KumoRFM, _remap_v2_1_checkpoint
from sdm.processing import EnsembleProcessor, TableDispatch
from sdm.testing import withCUDA


def _features_for_route(
    route: Literal["task", "related"],
) -> EnsembleProcessor:
    features = copy.deepcopy(KumoRFM.default_recipe().features)
    for module in features.modules():
        if isinstance(module, TableDispatch):
            module._route = route
    return features


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
            columns={"numerical": ("target",)},
            numerical=torch.randn(4, 1, device=device),
        )
    else:
        y = TableTensor(
            columns={"categorical": ("target",)},
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
        columns={Stype.categorical: ("target",)},
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
    model = _KumoRFM(
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


def test_default_recipe_preserves_ids() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: ("entity_id",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        id=ColumnarTensor((torch.tensor([10, 11]),)),
    )

    transformed = _features_for_route("task").fit_transform(table)

    assert transformed.columns[Stype.id] == ("entity_id",)
    assert transformed.id.equal(table.id)


def test_default_recipe_adds_calendar_fields_only_for_related() -> None:
    timestamp = int(
        datetime(2024, 2, 29, 23, 59, tzinfo=UTC).timestamp() * 1_000_000
    )
    table = TableTensor(
        columns={
            Stype.datetime: ("event_time",),
            Stype.id: ("entity_id",),
        },
        datetime=torch.tensor([[timestamp]], dtype=torch.int64),
        id=ColumnarTensor((torch.tensor([10]),)),
    )

    related = _features_for_route("related").fit_transform(table)
    task = _features_for_route("task").fit_transform(table)

    assert related.columns[Stype.datetime] == ("event_time",)
    assert any(
        name.startswith("event_time__")
        for name in related.columns[Stype.numerical]
    )
    assert task.columns[Stype.datetime] == ("event_time",)
    assert task.columns[Stype.numerical] == ()
    assert task.datetime.equal(table.datetime)
