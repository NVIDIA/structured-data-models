from typing import Any
from unittest.mock import Mock

import pytest
import torch
from sdm.models.kumorfm import KumoRFM
from sdm.models.kumorfm import model as model_module
from sdm.tensor.related_tables import HomogeneousGraph
from sdm.testing import onlyCUDA
from torch import Tensor


def _model(num_classes: int = 4) -> KumoRFM:
    return KumoRFM(
        num_classes=num_classes,
        num_quantiles=0,
        cell_channels=4,
        channels=8,
        num_row_layers=1,
        num_row_heads=1,
        group_size=1,
        num_inducing_points=2,
        num_icl_layers=1,
        num_icl_heads=2,
        dst_chunk_size=None,
    )


def _graph() -> HomogeneousGraph:
    return HomogeneousGraph(
        num_rows=12,
        num_task_rows=6,
        node_offsets={"native": 0, "readout": 4, "inactive": 10},
        edge_index=torch.tensor([[0, 10, 5, 11, 4], [4, 4, 7, 10, 8]]),
        edge_type=torch.tensor([0, 1, 2, 3, 0]),
        num_edge_types=4,
        task_edge_indices={},
        node_batch=torch.tensor([1, 0, 2, 3, 2, 0, 4, 1, 3, 5, -1, -1]),
        num_hops=1,
    )


def _table_hops(
    requires_grad: bool = False,
) -> tuple[dict[str, list[Tensor]], list[Tensor]]:
    bases = [
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.arange(18, dtype=torch.float32).reshape(6, 3) + 20,
        torch.arange(6, dtype=torch.float32).reshape(2, 3) + 50,
    ]
    if requires_grad:
        bases = [value.requires_grad_() for value in bases]
    native, readout, inactive = bases
    return {
        "readout": list(readout.tensor_split(2)),
        "inactive": [inactive],
        "native": list(native.tensor_split(2)),
    }, bases


def _forward(
    model: KumoRFM,
    table_hops: dict[str, list[Tensor]],
    *,
    graph: HomogeneousGraph | None = None,
    y: Tensor | None = None,
    y_lp: Tensor | None = None,
    context: Tensor | None = None,
    candidates: Tensor | None = None,
    readout_table: str = "readout",
    max_lp_context_size: int | None = None,
    max_train: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    context = torch.tensor([7, 5]) if context is None else context
    candidates = (
        torch.tensor([9, 4, 8, 6]) if candidates is None else candidates
    )
    return model.forward_link_prediction(
        table_hops,
        torch.tensor([2, 1]) if y is None else y,
        torch.tensor([0, 1]) if y_lp is None else y_lp,
        graph=_graph() if graph is None else graph,
        readout_table=readout_table,
        context_node_index=context,
        candidate_node_index=candidates,
        max_lp_context_size=max_lp_context_size,
        max_train=max_train,
        generator=generator,
    )


def test_api_injection_shared_graph_path_gradients_and_no_mutation() -> None:
    model = _model()
    table_hops, bases = _table_hops(requires_grad=True)
    y, y_lp = torch.tensor([2, 1]), torch.tensor([0, 1])
    context = torch.tensor([7, 5])
    candidates = torch.tensor([9, 4, 8, 6])
    graph = _graph()
    tensors = [
        *(part for parts in table_hops.values() for part in parts),
        y,
        y_lp,
        context,
        candidates,
        graph.edge_index,
        graph.edge_type,
        graph.node_batch,
    ]
    snapshot = [(value, value.clone(), value._version) for value in tensors]
    node_targets: list[Tensor] = []
    row_targets: list[Tensor] = []
    gnn_kwargs: list[dict[str, object]] = []
    model.table_hop_encoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: node_targets.append(
            kwargs["node_y"].clone()
        ),
        with_kwargs=True,
    )
    model.table_hop_encoder.row_embedding.register_forward_pre_hook(
        lambda _module, args: row_targets.append(args[1].detach().clone())
    )
    model.gnn.register_forward_pre_hook(
        lambda _module, _args, kwargs: gnn_kwargs.append(kwargs),
        with_kwargs=True,
    )

    out = _forward(
        model,
        table_hops,
        y=y,
        y_lp=y_lp,
        context=context,
        candidates=candidates,
        graph=graph,
        generator=torch.Generator().manual_seed(5),
    )
    out.square().sum().backward()

    assert out.size() == (4, 2)
    torch.testing.assert_close(
        node_targets[0],
        torch.tensor([1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    )
    assert [x.tolist() for x in row_targets] == [[1, 2], [], [1], [0]]
    torch.testing.assert_close(
        gnn_kwargs[0]["edge_index"],
        torch.tensor([[0, 5, 4], [4, 7, 8]]),
    )
    torch.testing.assert_close(
        gnn_kwargs[0]["edge_type"], torch.tensor([0, 2, 0])
    )
    assert bases[0].grad is not None
    assert bases[1].grad is not None
    assert model.head[2].weight.grad is not None
    for value, original, version in snapshot:
        torch.testing.assert_close(value, original)
        assert value._version == version


def test_canonical_context_cap_candidate_order_and_uncapped_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    table_hops, _ = _table_hops()
    captured: list[tuple[Tensor, Tensor]] = []

    def fake_encode(*_args: object, **_kwargs: object):
        rows = torch.arange(12, dtype=torch.float32).unsqueeze(1)
        return rows.expand(-1, 8), torch.ones(12, dtype=torch.bool)

    def fake_icl(x: Tensor, y: Tensor, **_kwargs: object) -> Tensor:
        captured.append((x.clone(), y.clone()))
        return x[y.numel() :]

    monkeypatch.setattr(model, "_encode_graph", fake_encode)
    monkeypatch.setattr(model.icl_block, "forward", fake_icl)
    model.head = torch.nn.Sequential(torch.nn.Identity())

    def capped_run() -> Tensor:
        with torch.inference_mode():
            return _forward(
                model,
                table_hops,
                max_lp_context_size=1,
                generator=torch.Generator().manual_seed(13),
            )

    expected_generator = torch.Generator().manual_seed(13)
    expected_index = torch.randperm(2, generator=expected_generator)[:1]
    first, repeated = capped_run(), capped_run()
    torch.testing.assert_close(repeated, first)
    expected_context = torch.tensor([5.0, 7.0])[expected_index]
    expected_y = torch.tensor([1, 0])[expected_index]
    torch.testing.assert_close(captured[0][0][0, 0], expected_context[0])
    torch.testing.assert_close(captured[0][1], expected_y)
    torch.testing.assert_close(first[:, 0], torch.tensor([9.0, 4.0, 8.0, 6.0]))

    generator = torch.Generator().manual_seed(17)
    state = generator.get_state().clone()
    with torch.inference_mode():
        _forward(model, table_hops, max_lp_context_size=2, generator=generator)
    torch.testing.assert_close(generator.get_state(), state)


def test_checkpointed_chunks_match_outputs_gradients_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    whole_model, chunked_model = _model().eval(), _model().eval()
    chunked_model.load_state_dict(whole_model.state_dict())
    whole_hops, whole_bases = _table_hops(requires_grad=True)
    chunked_hops, chunked_bases = _table_hops(requires_grad=True)
    whole_gen = torch.Generator().manual_seed(19)
    chunk_gen = torch.Generator().manual_seed(19)

    whole = _forward(whole_model, whole_hops, generator=whole_gen)
    whole.square().sum().backward()
    checkpoint_spy = Mock(wraps=model_module.checkpoint)
    monkeypatch.setattr(model_module, "checkpoint", checkpoint_spy)
    monkeypatch.setattr(model_module, "_LINK_PREDICTION_CHUNK_SIZE", 2)
    chunked = _forward(chunked_model, chunked_hops, generator=chunk_gen)
    chunked.square().sum().backward()

    assert checkpoint_spy.call_count == 2
    assert all(
        call.kwargs["use_reentrant"] is False
        for call in checkpoint_spy.call_args_list
    )
    torch.testing.assert_close(chunked, whole)
    torch.testing.assert_close(chunk_gen.get_state(), whole_gen.get_state())
    for left, right in zip(whole_bases, chunked_bases):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None and right.grad is not None:
            torch.testing.assert_close(right.grad, left.grad)
    chunked_parameters = dict(chunked_model.named_parameters())
    for name, parameter in whole_model.named_parameters():
        other = chunked_parameters[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None and other.grad is not None:
            torch.testing.assert_close(other.grad, parameter.grad)


def test_empty_candidates_return_graph_connected_two_logits() -> None:
    model = _model()
    table_hops, _ = _table_hops(requires_grad=True)
    out = _forward(
        model,
        table_hops,
        candidates=torch.empty(0, dtype=torch.long),
        generator=torch.Generator().manual_seed(3),
    )
    assert out.size() == (0, 2)
    out.sum().backward()
    assert model.head[2].weight.grad is not None


@pytest.mark.parametrize(
    ("num_classes", "overrides"),
    [
        (1, {}),
        (4, {"readout_table": "missing"}),
        (4, {"y": torch.tensor([2.0, 1.0])}),
        (4, {"y": torch.tensor([4, 1])}),
        (4, {"y_lp": torch.tensor([0.0, 1.0])}),
        (4, {"y_lp": torch.tensor([0, 2])}),
        (4, {"y_lp": torch.tensor([0])}),
        (4, {"context": torch.tensor([5]), "y_lp": torch.tensor([1])}),
        (
            4,
            {
                "context": torch.tensor([7, 5, 4]),
                "y_lp": torch.tensor([0, 1, 1]),
            },
        ),
        (4, {"context": torch.tensor([5, 5])}),
        (4, {"context": torch.tensor([5, 12])}),
        (4, {"candidates": torch.tensor([4, 4])}),
        (4, {"candidates": torch.tensor([12])}),
        (4, {"candidates": torch.tensor([1])}),
        (4, {"candidates": torch.tensor([5])}),
        (4, {"max_train": 0}),
        (4, {"max_lp_context_size": 0}),
        (
            4,
            {
                "graph": _graph()._replace(
                    node_batch=torch.tensor(
                        [1, 0, 2, 3, -1, 0, 4, 1, 3, 5, -1, -1]
                    )
                ),
                "candidates": torch.tensor([4]),
            },
        ),
        (
            4,
            {
                "graph": _graph()._replace(
                    node_batch=torch.tensor(
                        [1, 0, 2, 3, 2, 2, 4, 3, 3, 5, -1, -1]
                    )
                ),
                "context": torch.empty(0, dtype=torch.long),
                "y_lp": torch.empty(0, dtype=torch.long),
            },
        ),
    ],
)
def test_link_validation_precedes_encoding_and_rng(
    num_classes: int,
    overrides: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(num_classes)
    monkeypatch.setattr(
        model,
        "_encode_graph",
        lambda *_args, **_kwargs: pytest.fail("encoding started"),
    )
    generator = torch.Generator().manual_seed(23)
    state = generator.get_state().clone()
    table_hops, _ = _table_hops()
    with pytest.raises((TypeError, ValueError)):
        _forward(model, table_hops, generator=generator, **overrides)
    torch.testing.assert_close(generator.get_state(), state)


def test_table_hop_preflight_precedes_row_model_and_rng() -> None:
    model = _model()
    table_hops, _ = _table_hops()
    table_hops.pop("inactive")
    model.table_hop_encoder.row_embedding.register_forward_pre_hook(
        lambda *_args: pytest.fail("row model started")
    )
    generator = torch.Generator().manual_seed(29)
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match="keys"):
        _forward(model, table_hops, generator=generator)
    torch.testing.assert_close(generator.get_state(), state)


@onlyCUDA
def test_generator_device_is_validated_before_encoding() -> None:
    model = _model()
    model.__dict__["_encode_graph"] = Mock(side_effect=AssertionError())
    generator = torch.Generator(device="cuda")
    state = generator.get_state().clone()
    table_hops, _ = _table_hops()
    with pytest.raises(ValueError, match="graph device"):
        _forward(model, table_hops, generator=generator)
    torch.testing.assert_close(generator.get_state(), state)
