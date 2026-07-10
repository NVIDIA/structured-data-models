import inspect

import pytest
import torch
from sdm import ColumnarTensor, RelatedTables, Relationship, TableTensor
from sdm.cache import Cache
from sdm.models import KumoRFM
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.model import _KumoRFM
from sdm.models.kumorfm.table_hop_encoder import (
    TableHopEncoder,
    _TableHopEncoding,
)
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.relational.sampler import EXAMPLE_ID
from test.models.kumorfm.sample_utils import with_full_sample
from torch import Tensor


class _StubCore(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        related_tables: RelatedTables | None,
        *,
        cache: Cache | None = None,
    ) -> Tensor:
        del related_tables
        del cache
        self.calls += 1
        return x.new_full((x.size(0) - y.size(0), 1), self.value)


def _core(num_classes: int, num_quantiles: int) -> _KumoRFM:
    return _KumoRFM(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
        cell_channels=4,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=1,
        num_inducing_points=2,
        group_size=1,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
    )


def _table(
    *,
    example: list[int],
    ids: dict[str, list[int]],
    value: list[float],
) -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("value",),
            "id": (EXAMPLE_ID, *ids),
        },
        numerical=torch.tensor(value).unsqueeze(-1),
        id=ColumnarTensor(
            (
                torch.tensor(example),
                *(torch.tensor(ids[column]) for column in ids),
            )
        ),
    )


def _related_tables(
    *,
    entity_example: list[int] | None = None,
    one_hop: bool = False,
) -> RelatedTables:
    entity_example = entity_example or [0, 1, 2, 3]
    entity_id = [100 + example for example in entity_example]
    tables = {
        "entity": _table(
            example=entity_example,
            ids={"entity_id": entity_id},
            value=[10.0 + example for example in entity_example],
        )
    }
    relationships: tuple[Relationship, ...] = ()
    if one_hop:
        tables["events"] = _table(
            example=[0, 2, 3],
            ids={
                "event_id": [1, 2, 3],
                "missing_entity_id": [900, 902, 903],
                "entity_id": [100, 102, 103],
            },
            value=[4.0, 5.0, 6.0],
        )
        relationships = (
            Relationship(
                left_table="events",
                left_columns=(EXAMPLE_ID, "missing_entity_id"),
                right_table="entity",
                right_columns=(EXAMPLE_ID, "entity_id"),
            ),
            Relationship(
                left_table="events",
                left_columns=(EXAMPLE_ID, "entity_id"),
                right_table="entity",
                right_columns=(EXAMPLE_ID, "entity_id"),
            ),
        )

    related_tables = RelatedTables(
        tables=tables,
        relationships=relationships,
        task_links=(
            {
                "task_columns": (EXAMPLE_ID, "entity_id"),
                "table": "entity",
                "table_columns": (EXAMPLE_ID, "entity_id"),
            },
        ),
    )
    if one_hop:
        return with_full_sample(
            related_tables,
            num_task_rows=len(entity_example),
        )
    return related_tables


def _x() -> Tensor:
    return torch.arange(8, dtype=torch.float).view(4, 2)


def test_public_model_composition_and_target_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(KumoRFM).parameters
    model = KumoRFM(pretrained=False, device="meta")

    assert parameters["pretrained"].default is True
    assert isinstance(model.cls_model, _KumoRFM)
    assert isinstance(model.reg_model, _KumoRFM)
    assert isinstance(model.cls_model.table_hop_encoder, TableHopEncoder)
    assert isinstance(
        model.cls_model.table_hop_encoder.row_embedding,
        RowEmbedding,
    )
    assert isinstance(model.cls_model.gnn, InvariantGNN)
    assert isinstance(model.cls_model.icl_block, ICLBlock)
    assert model.cls_model.head[-1].out_features == 10
    assert model.reg_model.head[-1].out_features == 999
    assert len(model.cls_model.icl_block.layers) == 12
    assert not model.training

    cls_model = _StubCore(1.0)
    reg_model = _StubCore(2.0)
    monkeypatch.setattr(model, "cls_model", cls_model)
    monkeypatch.setattr(model, "reg_model", reg_model)

    cls_out = model._forward(_x(), torch.tensor([0, 1]), None, None)
    reg_out = model._forward(_x(), torch.tensor([0.0, 1.0]), None, None)

    assert torch.equal(cls_out, torch.ones(2, 1))
    assert torch.equal(reg_out, torch.full((2, 1), 2.0))
    assert cls_model.calls == 1
    assert reg_model.calls == 1


def test_unavailable_public_features() -> None:
    with pytest.raises(NotImplementedError, match="checkpoints"):
        KumoRFM(pretrained=True)
    with pytest.raises(NotImplementedError):
        KumoRFM.default_recipe()


@pytest.mark.parametrize(
    ("y", "output_channels"),
    [
        (torch.tensor([0, 1], dtype=torch.int16), 3),
        (torch.tensor([0.25, -0.5], dtype=torch.float64), 5),
    ],
)
def test_public_forward_is_inference_safe(
    y: Tensor,
    output_channels: int,
) -> None:
    model = KumoRFM(pretrained=False, device="meta")
    model.cls_model = _core(num_classes=3, num_quantiles=0)
    model.reg_model = _core(num_classes=0, num_quantiles=5)

    out = model(_x(), y, _related_tables(one_hop=True))

    assert out.size() == (2, output_channels)
    assert torch.is_inference(out)
    assert torch.isfinite(out).all()


def test_multiple_estimators_are_explicitly_deferred() -> None:
    model = KumoRFM(pretrained=False, device="meta")

    with pytest.raises(ValueError, match="one estimator"):
        model(_x(), torch.tensor([0, 1]), num_estimators=2)
    with pytest.raises(ValueError, match="one estimator"):
        model.fit(_x(), torch.tensor([0, 1]), num_estimators=2)


@pytest.mark.parametrize(
    ("num_classes", "num_quantiles", "y", "output_channels"),
    [
        (3, 0, torch.tensor([0, 1]), 3),
        (0, 5, torch.tensor([0.25, -0.5]), 5),
    ],
)
def test_zero_hop_classification_and_regression(
    num_classes: int,
    num_quantiles: int,
    y: Tensor,
    output_channels: int,
) -> None:
    model = _core(num_classes, num_quantiles)

    out = model(_x(), y, _related_tables())

    assert out.size() == (2, output_channels)
    assert torch.isfinite(out).all()


def test_one_hop_exact_graph_forward() -> None:
    model = _core(num_classes=3, num_quantiles=0)
    gnn_inputs: dict[str, object] = {}

    def record_gnn_input(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        gnn_inputs.update(kwargs)

    handle = model.gnn.register_forward_pre_hook(
        record_gnn_input,
        with_kwargs=True,
    )
    out = model(
        _x(),
        torch.tensor([0, 1]),
        _related_tables(one_hop=True),
    )
    handle.remove()

    assert out.size() == (2, 3)
    assert torch.isfinite(out).all()
    assert gnn_inputs["num_hops"] == 1
    edge_index_dict = gnn_inputs["edge_index_dict"]
    assert isinstance(edge_index_dict, dict)
    assert list(edge_index_dict) == [("events", "1", "entity")]


def test_positive_hop_without_retained_edges_still_runs_gnn() -> None:
    model = InvariantGNN(channels=8)
    x = torch.randn(4, 8)

    out = model(
        x_dict={"entity": x},
        edge_index_dict={},
        readout_table="entity",
        num_hops=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert out is not x
    assert out.size() == x.size()
    assert torch.isfinite(out).all()


def test_roots_are_selected_after_gnn_in_task_order() -> None:
    model = _core(num_classes=3, num_quantiles=0)
    encodings: list[_TableHopEncoding] = []
    gnn_outputs: list[Tensor] = []
    icl_inputs: list[Tensor] = []

    def record_encoding(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        output: _TableHopEncoding,
    ) -> None:
        encodings.append(output)

    def record_gnn_output(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        output: Tensor,
    ) -> None:
        gnn_outputs.append(output.detach().clone())

    def record_icl_input(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        value = kwargs["x"]
        assert isinstance(value, Tensor)
        icl_inputs.append(value.detach().clone())

    encoding_handle = model.table_hop_encoder.register_forward_hook(
        record_encoding
    )
    gnn_handle = model.gnn.register_forward_hook(record_gnn_output)
    icl_handle = model.icl_block.register_forward_pre_hook(
        record_icl_input,
        with_kwargs=True,
    )
    out = model(
        _x(),
        torch.tensor([0, 1]),
        _related_tables(entity_example=[2, 0, 3, 1]),
    )
    encoding_handle.remove()
    gnn_handle.remove()
    icl_handle.remove()

    assert encodings[0].root_index.tolist() == [1, 3, 0, 2]
    torch.testing.assert_close(
        icl_inputs[0],
        gnn_outputs[0].index_select(0, encodings[0].root_index),
    )
    assert out.size() == (2, 3)


@pytest.mark.parametrize(
    ("model", "y", "error", "message"),
    [
        (
            _core(num_classes=3, num_quantiles=0),
            torch.tensor([], dtype=torch.long),
            ValueError,
            "at least one context",
        ),
        (
            _core(num_classes=3, num_quantiles=0),
            torch.tensor([0.0, 1.0]),
            TypeError,
            "integers",
        ),
        (
            _core(num_classes=3, num_quantiles=0),
            torch.tensor([False, True]),
            TypeError,
            "integers",
        ),
        (
            _core(num_classes=3, num_quantiles=0),
            torch.tensor([0, 3]),
            ValueError,
            "between 0 and 2",
        ),
        (
            _core(num_classes=0, num_quantiles=5),
            torch.tensor([0, 1]),
            TypeError,
            "floating point",
        ),
    ],
)
def test_target_validation(
    model: _KumoRFM,
    y: Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        model(_x(), y, _related_tables())
