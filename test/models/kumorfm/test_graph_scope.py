import copy

import pytest
import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models.kumorfm.graph import HomogeneousGraph, LayeredGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.testing import withCUDA


def _graph(
    case: str,
    device: torch.device,
) -> tuple[HomogeneousGraph, Tensor]:
    if case == "empty":
        num_nodes = 7
        edges: list[tuple[int, int]] = []
        readout_index = torch.tensor([5, 1, 3], device=device)
        target_start = 0
    elif case == "chain":
        num_nodes = 8
        edges = [(i, i + 1) for i in range(num_nodes - 1)]
        readout_index = torch.tensor([7, 5], device=device)
        target_start = 0
    elif case == "cycle":
        num_nodes = 8
        edges = [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 0),
            (0, 3),
            (4, 5),
            (5, 4),
        ]
        readout_index = torch.tensor([3, 0, 3], device=device)
        target_start = 0
    elif case == "dense":
        num_nodes = 7
        edges = [
            (src, dst) for src in range(num_nodes) for dst in range(num_nodes)
        ]
        readout_index = torch.tensor([6, 2], device=device)
        target_start = 0
    elif case == "dense_saturation":
        num_nodes = 6
        edges = [
            (src, dst) for src in range(num_nodes) for dst in range(num_nodes)
        ]
        readout_index = torch.tensor([5, 1, 3, 0, 4, 2], device=device)
        target_start = 0
    elif case == "offset_readout":
        num_nodes = 9
        edges = [
            (0, 3),
            (3, 0),
            (1, 5),
            (5, 1),
            (2, 7),
            (7, 2),
            (5, 6),
            (6, 5),
            (6, 7),
            (7, 6),
            (7, 8),
            (8, 7),
        ]
        readout_index = torch.tensor([5, 1, 3], device=device)
        target_start = 3
    else:
        raise ValueError(case)

    if len(edges) == 0:
        row = torch.empty(0, dtype=torch.long, device=device)
        col = torch.empty(0, dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
            device=device,
        ).t()
        row, col = edge_index

    edge_type = torch.arange(
        row.numel(),
        dtype=torch.long,
        device=device,
    ).remainder(4)
    col, perm = col.sort(stable=True)
    row = row[perm]
    edge_type = edge_type[perm]
    colptr = torch._convert_indices_from_coo_to_csr(col, num_nodes)
    return (
        HomogeneousGraph(
            row=row,
            col=col,
            colptr=colptr,
            edge_type=edge_type,
            num_edge_types=4,
            start_node_offsets={"prefix": 0, "target": target_start},
            end_node_offsets={"prefix": target_start, "target": num_nodes},
        ),
        readout_index,
    )


def _layered_graph(
    graph: HomogeneousGraph,
    readout_index: Tensor,
    num_hops: int,
    *,
    readout_scoped: bool,
) -> LayeredGraph:
    readout_index = readout_index + graph.start_node_offsets["target"]
    if not readout_scoped:
        return graph.full_layered(
            num_layers=num_hops,
            readout_index=readout_index,
        )

    active = torch.zeros(
        graph.num_nodes,
        dtype=torch.bool,
        device=graph.row.device,
    )
    active[readout_index] = True
    node_masks = [active]
    for _ in range(num_hops):
        previous = active.clone()
        previous[graph.row[active[graph.col]]] = True
        node_masks.append(previous)
        active = previous
    return graph.layered(
        node_masks=node_masks,
        readout_index=readout_index,
    )


def _cache(edge_type_emb: Tensor) -> Cache:
    return Cache({"edge_type_emb": edge_type_emb}).freeze()


@withCUDA
@pytest.mark.parametrize(
    ("case", "num_hops"),
    [
        ("empty", 3),
        ("chain", 3),
        ("cycle", 0),
        ("cycle", 4),
        ("dense", 3),
        ("dense_saturation", 2),
        ("offset_readout", 3),
    ],
)
def test_layered_graph_preserves_output(
    case: str,
    num_hops: int,
    device: torch.device,
) -> None:
    graph, readout_index = _graph(case, device)
    model = InvariantGNN(channels=8, device=device)
    x = torch.randn(graph.num_nodes, 8, device=device)
    edge_type_emb = torch.randn(graph.num_edge_types, 8, device=device)

    full = model(
        x=x,
        graph=_layered_graph(
            graph,
            readout_index,
            num_hops,
            readout_scoped=False,
        ),
        cache=_cache(edge_type_emb),
    )
    readout = model(
        x=x,
        graph=_layered_graph(
            graph,
            readout_index,
            num_hops,
            readout_scoped=True,
        ),
        cache=_cache(edge_type_emb),
    )

    torch.testing.assert_close(readout, full, rtol=5e-3, atol=3e-5)


def test_readout_scope_reduces_message_passing_work() -> None:
    graph, readout_index = _graph("chain", torch.device("cpu"))
    full = _layered_graph(
        graph,
        readout_index,
        num_hops=2,
        readout_scoped=False,
    )
    readout = _layered_graph(
        graph,
        readout_index,
        num_hops=2,
        readout_scoped=True,
    )

    assert sum(layer.row.numel() for layer in readout.layers) < sum(
        layer.row.numel() for layer in full.layers
    )
    assert sum(layer.colptr.numel() - 1 for layer in readout.layers) < sum(
        layer.colptr.numel() - 1 for layer in full.layers
    )


def test_layered_graph_preserves_float64_gradients() -> None:
    graph, readout_index = _graph("cycle", torch.device("cpu"))
    full_model = InvariantGNN(channels=6, dtype=torch.float64)
    readout_model = copy.deepcopy(full_model)
    full_x = torch.randn(
        graph.num_nodes,
        6,
        dtype=torch.float64,
        requires_grad=True,
    )
    readout_x = full_x.detach().clone().requires_grad_()
    full_generator = torch.Generator().manual_seed(0)
    readout_generator = torch.Generator().manual_seed(0)

    full = full_model(
        x=full_x,
        graph=_layered_graph(
            graph,
            readout_index,
            num_hops=3,
            readout_scoped=False,
        ),
        generator=full_generator,
    )
    readout = readout_model(
        x=readout_x,
        graph=_layered_graph(
            graph,
            readout_index,
            num_hops=3,
            readout_scoped=True,
        ),
        generator=readout_generator,
    )
    weight = torch.randn_like(full)
    (full * weight).sum().backward()
    (readout * weight).sum().backward()

    torch.testing.assert_close(readout, full)
    torch.testing.assert_close(readout_x.grad, full_x.grad)
    for readout_parameter, full_parameter in zip(
        readout_model.parameters(),
        full_model.parameters(),
        strict=True,
    ):
        if full_parameter.grad is None:
            assert readout_parameter.grad is None
        else:
            torch.testing.assert_close(
                readout_parameter.grad,
                full_parameter.grad,
            )
