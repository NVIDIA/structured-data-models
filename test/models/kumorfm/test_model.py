import inspect

import pytest
import torch
from sdm.models import KumoRFM
from sdm.models import kumorfm as kumorfm_package
from sdm.models.kumorfm import TableHopEncoder
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import InvariantGNN
from sdm.tensor.related_tables import HomogeneousGraph
from torch import Tensor


def _model(num_classes: int = 3, num_quantiles: int = 0) -> KumoRFM:
    return KumoRFM(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
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
        num_rows=6,
        num_task_rows=4,
        node_offsets={"entities": 0, "facts": 4},
        edge_index=torch.tensor([[4, 0, 5, 1], [0, 4, 1, 5]]),
        edge_type=torch.tensor([0, 1, 0, 1]),
        num_edge_types=2,
        task_edge_indices={7: torch.tensor([[2, 0, 3, 1], [2, 0, 3, 1]])},
        node_batch=torch.tensor([0, 1, 2, 3, 0, 1]),
        num_hops=1,
    )


def _table_hops(
    *,
    num_parts: int = 2,
    requires_grad: bool = False,
) -> tuple[dict[str, list[Tensor]], Tensor, Tensor]:
    entities = torch.randn(4, 3, requires_grad=requires_grad)
    facts = torch.randn(2, 2, requires_grad=requires_grad)
    entity_parts = (
        [entities]
        if num_parts == 1
        else list(entities.tensor_split(num_parts))
    )
    fact_parts = (
        [facts] if num_parts == 1 else list(facts.tensor_split(num_parts))
    )
    return {"entities": entity_parts, "facts": fact_parts}, entities, facts


def test_architecture_reuses_generic_components_and_owns_state_once() -> None:
    parameters = inspect.signature(KumoRFM.forward).parameters
    expected = (
        "self table_hops y graph entity_relationship max_train generator"
    )
    assert " ".join(parameters) == expected
    assert parameters["graph"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        parameters["entity_relationship"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert kumorfm_package.__all__ == ["TableHopEncoder", "KumoRFM"]
    model = _model()
    row_embedding = model.table_hop_encoder.row_embedding
    assert isinstance(model.table_hop_encoder, TableHopEncoder)
    assert isinstance(row_embedding, RowEmbedding)
    assert isinstance(model.gnn, InvariantGNN)
    assert isinstance(model.icl_block, ICLBlock)
    assert not hasattr(model, "row_embedding")
    assert [
        name
        for name, module in model.named_modules()
        if isinstance(module, RowEmbedding)
    ] == ["table_hop_encoder.row_embedding"]
    assert isinstance(model.head[0], torch.nn.Linear)
    assert isinstance(model.head[1], torch.nn.GELU)
    assert isinstance(model.head[2], torch.nn.Linear)
    assert model.head[2].out_features == 3


@pytest.mark.parametrize(
    ("num_classes", "num_quantiles", "y", "width"),
    [
        (3, 0, torch.tensor([0, 2]), 3),
        (0, 5, torch.tensor([0.25, -0.5]), 5),
    ],
)
def test_output_shapes_and_gradients(
    num_classes: int,
    num_quantiles: int,
    y: Tensor,
    width: int,
) -> None:
    table_hops, entities, facts = _table_hops(requires_grad=True)
    model = _model(num_classes=num_classes, num_quantiles=num_quantiles)
    out = model(
        table_hops,
        y,
        graph=_graph(),
        entity_relationship=7,
        max_train=1,
        generator=torch.Generator().manual_seed(11),
    )
    out.square().mean().backward()
    assert out.size() == (2, width)
    assert entities.grad is not None
    assert facts.grad is not None
    assert model.table_hop_encoder.row_embedding.lin.weight.grad is not None
    assert model.gnn.src_lin.weight.grad is not None
    assert any(p.grad is not None for p in model.icl_block.parameters())
    assert model.head[2].weight.grad is not None


def test_zero_hop_sorts_shuffled_roots() -> None:
    table_hops, _, _ = _table_hops(num_parts=1)
    model = _model().eval()
    encoded: list[Tensor] = []
    icl_input: list[Tensor] = []
    model.table_hop_encoder.register_forward_hook(
        lambda _module, _args, output: encoded.append(output[0])
    )
    model.icl_block.register_forward_pre_hook(
        lambda _module, args: icl_input.append(args[0])
    )
    with torch.inference_mode():
        model(
            table_hops,
            torch.tensor([0, 1]),
            graph=_graph(),
            entity_relationship=7,
        )
    torch.testing.assert_close(icl_input[0], encoded[0][:4])


def test_filters_inactive_edges_and_shares_generator() -> None:
    graph = HomogeneousGraph(
        num_rows=5,
        num_task_rows=3,
        node_offsets={"entities": 0, "inactive": 3},
        edge_index=torch.tensor([[0, 3, 1, 4, 0], [1, 1, 2, 3, 2]]),
        edge_type=torch.tensor([4, 3, 1, 2, 4]),
        num_edge_types=5,
        task_edge_indices={7: torch.tensor([[2, 0, 1], [2, 0, 1]])},
        node_batch=torch.tensor([0, 1, 2, -1, -1]),
        num_hops=2,
    )
    table_hops = {
        "entities": list(torch.randn(3, 2).tensor_split(3)),
        "inactive": [torch.randn(2, 2)],
    }
    model = _model().eval()
    encoder_kwargs: list[dict[str, object]] = []
    gnn_kwargs: list[dict[str, object]] = []
    model.table_hop_encoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: encoder_kwargs.append(kwargs),
        with_kwargs=True,
    )
    model.gnn.register_forward_pre_hook(
        lambda _module, _args, kwargs: gnn_kwargs.append(kwargs),
        with_kwargs=True,
    )
    generator = torch.Generator().manual_seed(17)
    model(
        table_hops,
        torch.tensor([0]),
        graph=graph,
        entity_relationship=7,
        generator=generator,
    )
    torch.testing.assert_close(
        gnn_kwargs[0]["edge_index"],
        torch.tensor([[0, 1, 0], [1, 2, 2]]),
    )
    torch.testing.assert_close(
        gnn_kwargs[0]["edge_type"], torch.tensor([4, 1, 4])
    )
    assert gnn_kwargs[0]["num_edge_types"] == 5
    assert gnn_kwargs[0]["num_hops"] == 2
    assert encoder_kwargs[0]["generator"] is generator
    assert gnn_kwargs[0]["generator"] is generator


def test_generator_is_deterministic() -> None:
    table_hops, _, _ = _table_hops()
    model = _model().eval()

    def run() -> Tensor:
        return model(
            table_hops,
            torch.tensor([0, 1]),
            graph=_graph(),
            entity_relationship=7,
            max_train=1,
            generator=torch.Generator().manual_seed(23),
        )

    with torch.inference_mode():
        first, repeated = run(), run()
    torch.testing.assert_close(repeated, first)


def _validation_graph() -> HomogeneousGraph:
    return HomogeneousGraph(
        num_rows=4,
        num_task_rows=3,
        node_offsets={"table": 0},
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty(0, dtype=torch.long),
        num_edge_types=0,
        task_edge_indices={7: torch.tensor([[2, 0, 1], [2, 0, 1]])},
        node_batch=torch.tensor([0, 1, 2, -1]),
        num_hops=0,
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_relationship", "absent"),
        ("task_ids", "each task ID once"),
        ("duplicate_roots", "unique"),
        ("root_range", "out-of-range"),
        ("root_batch", "node_batch IDs"),
        ("assigned_id", "below num_task_rows"),
        ("empty_y", "nonempty"),
        ("matrix_y", "one-dimensional"),
        ("long_y", "more values"),
        ("device_y", "graph device"),
        ("class_dtype", "integral"),
        ("class_range", r"\[0, 3\)"),
        ("regression_dtype", "floating"),
    ],
)
def test_rejects_invalid_roots_and_targets_before_model_work(
    case: str,
    match: str,
) -> None:
    graph = _validation_graph()
    relationship = 7
    y = torch.tensor([0])
    model = _model()
    if case == "missing_relationship":
        relationship = 8
    elif case == "task_ids":
        graph = graph._replace(
            task_edge_indices={7: torch.tensor([[0, 0, 2], [0, 1, 2]])}
        )
    elif case == "duplicate_roots":
        graph = graph._replace(
            task_edge_indices={7: torch.tensor([[0, 1, 2], [0, 0, 2]])}
        )
    elif case == "root_range":
        graph = graph._replace(
            task_edge_indices={7: torch.tensor([[0, 1, 2], [0, 1, 4]])}
        )
    elif case == "root_batch":
        graph = graph._replace(node_batch=torch.tensor([0, 2, 2, -1]))
    elif case == "assigned_id":
        graph = graph._replace(node_batch=torch.tensor([0, 1, 2, 3]))
    elif case == "empty_y":
        y = torch.empty(0, dtype=torch.long)
    elif case == "matrix_y":
        y = torch.tensor([[0]])
    elif case == "long_y":
        y = torch.tensor([0, 1, 2, 0])
    elif case == "device_y":
        y = torch.ones(1, dtype=torch.long, device="meta")
    elif case == "class_dtype":
        y = torch.tensor([0.0])
    elif case == "class_range":
        y = torch.tensor([3])
    else:
        model = _model(num_classes=0, num_quantiles=2)
    generator = torch.Generator().manual_seed(29)
    generator_state = generator.get_state().clone()
    handle = model.table_hop_encoder.register_forward_pre_hook(
        lambda *_args: pytest.fail("model work started")
    )
    with pytest.raises((TypeError, ValueError), match=match):
        model(
            {"table": [torch.randn(4, 2)]},
            y,
            graph=graph,
            entity_relationship=relationship,
            generator=generator,
        )
    handle.remove()
    torch.testing.assert_close(generator.get_state(), generator_state)
