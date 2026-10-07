"""Numerical gates for bounded-memory relational GNN inference."""

import pytest
import research.multigpu.blocked_gnn as blocked_gnn
import torch
from research.multigpu.blocked_gnn import (
    BlockedInvariantGNN,
    destination_statistics,
)

from sdm._kernels import segment_multi_reduce
from sdm.cache import Cache
from sdm.models.kumo.relational.graph import HomogeneousGraph
from sdm.models.kumo.relational.invariant_gnn import InvariantGNN


def make_graph(empty: bool = False, reorder: bool = False) -> HomogeneousGraph:
    # Isolated destinations, repeated edges, self edges, multiple edge types.
    row = torch.tensor([] if empty else [2, 2, 0, 4, 1, 3, 0, 5, 3, 2]).long()
    col = torch.tensor([] if empty else [0, 0, 0, 2, 2, 3, 3, 3, 5, 5]).long()
    edge_type = torch.arange(row.numel()) % 3
    if reorder:
        permutation = torch.tensor([2, 1, 0, 4, 3, 7, 6, 5, 9, 8])
        row, edge_type = row[permutation], edge_type[permutation]
    return HomogeneousGraph(
        row=row,
        col=col,
        colptr=torch._convert_indices_from_coo_to_csr(col, 7),
        edge_type=edge_type,
        num_edge_types=3,
        start_node_offsets={"context": 0, "target": 2},
        end_node_offsets={"context": 2, "target": 7},
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [1, 2, 4, 20])
@pytest.mark.parametrize(
    "case", ["random", "constant", "nonfinite", "empty", "reordered"]
)
def test_statistics_preserve_reduction(dtype, block_size, case):
    graph = make_graph(empty=case == "empty", reorder=case == "reordered")
    src = torch.randn(7, 9).to(dtype)
    edge_attr = torch.randn(3, 9).to(dtype)
    if case == "constant":
        src.fill_(0.5)
        edge_attr.zero_()
    if case == "nonfinite":
        src[2, 0] = float("nan")
        src[0, 1] = float("inf")
        src[3, 2] = -float("inf")
    expected = segment_multi_reduce(
        src, graph.row, edge_attr, graph.colptr, graph.edge_type
    )
    actual = torch.cat(
        [
            destination_statistics(
                src, graph, edge_attr, start, min(start + block_size, 7)
            )
            for start in range(0, 7, block_size)
        ]
    )
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [1, 3, 20])
@pytest.mark.parametrize("num_hops", [0, 1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_native_module_and_cache_parity(dtype, block_size, num_hops, empty):
    graph = make_graph(empty=empty)
    model = InvariantGNN(16, dtype=dtype).eval()
    candidate = BlockedInvariantGNN(model, block_size)
    x = torch.randn(7, 16).to(dtype)
    kwargs = {
        "graph": graph,
        "readout_table": "target",
        "readout_index": torch.tensor([4, 0, 2, 0]),
        "num_hops": num_hops,
    }
    tolerance = (
        {"rtol": 1e-5, "atol": 2e-6}
        if dtype == torch.float32
        else {"rtol": 2e-2, "atol": 2e-2}
    )
    # Edge-type random draws must match; this seed is part of the comparison.
    with torch.inference_mode():
        baseline_cache, candidate_cache = Cache(), Cache()
        expected = model(
            x,
            **kwargs,
            cache=baseline_cache,
            generator=torch.Generator().manual_seed(1729),
        )
        actual = candidate(
            x,
            **kwargs,
            cache=candidate_cache,
            generator=torch.Generator().manual_seed(1729),
        )
        torch.testing.assert_close(actual, expected, **tolerance)
        if num_hops:
            torch.testing.assert_close(
                candidate_cache["edge_type_emb"],
                baseline_cache["edge_type_emb"],
                rtol=0,
                atol=0,
            )
        baseline_cache.freeze()
        candidate_cache.freeze()
        query = x.flip(0)
        expected = model(query, **kwargs, cache=baseline_cache)
        actual = candidate(query, **kwargs, cache=candidate_cache)
        torch.testing.assert_close(actual, expected, **tolerance)


def test_rejects_training():
    model = BlockedInvariantGNN(InvariantGNN(4), 2)
    with pytest.raises(RuntimeError, match="inference_mode"):
        model(
            torch.randn(7, 4),
            make_graph(),
            readout_table="target",
            readout_index=torch.tensor([0]),
            num_hops=2,
        )


@pytest.mark.parametrize("block_size", [0, -1])
def test_rejects_nonpositive_block_size(block_size):
    with pytest.raises(ValueError, match="positive"):
        BlockedInvariantGNN(InvariantGNN(4), block_size)


@pytest.mark.parametrize("block_size", [1, 3, 20])
def test_fp32_parameters_bfloat16_autocast(block_size):
    model = InvariantGNN(16).eval()
    candidate = BlockedInvariantGNN(model, block_size)
    x = torch.randn(7, 16)
    kwargs = {
        "graph": make_graph(),
        "readout_table": "target",
        "readout_index": torch.tensor([4, 0, 2]),
        "num_hops": 2,
    }
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(
            x, **kwargs, generator=torch.Generator().manual_seed(1729)
        )
        actual = candidate(
            x, **kwargs, generator=torch.Generator().manual_seed(1729)
        )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_statistics_workspace_shape_is_bounded(monkeypatch):
    shapes = []

    def record_statistics(*args):
        output = destination_statistics(*args)
        shapes.append(output.shape)
        return output

    monkeypatch.setattr(
        blocked_gnn, "destination_statistics", record_statistics
    )
    model = BlockedInvariantGNN(InvariantGNN(16), block_size=3)
    with torch.inference_mode():
        model(
            torch.randn(7, 16),
            make_graph(),
            readout_table="target",
            readout_index=torch.tensor([0, 4]),
            num_hops=2,
        )
    assert max(shape[0] for shape in shapes) == 3
    assert sum(shape[0] for shape in shapes) == 2 * 7
    assert all(shape[1:] == (5, 16) for shape in shapes)
