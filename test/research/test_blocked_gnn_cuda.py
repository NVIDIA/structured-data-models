"""CUDA gates: absolute CSR slices, original Triton stats, and cached GNNs."""

from dataclasses import replace

import pytest
import torch
from research.multigpu.blocked_gnn import (
    BlockedInvariantGNN,
    destination_statistics,
)

from sdm._kernels import segment_multi_reduce
from sdm.cache import Cache
from sdm.models.kumo.relational.graph import HomogeneousGraph
from sdm.models.kumo.relational.invariant_gnn import InvariantGNN

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def make_cuda_graph(case: str = "random") -> HomogeneousGraph:
    nodes = 257
    counts = torch.arange(nodes) % 5
    if case == "empty":
        counts.zero_()
    col = torch.arange(nodes).repeat_interleave(counts)
    row = (torch.arange(col.numel()) * 7 + 3) % nodes
    edge_type = torch.arange(row.numel()) % 3
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    if case == "reordered":
        permutation = torch.cat(
            [
                torch.arange(offsets[i], offsets[i + 1]).flip(0)
                for i in range(nodes)
            ]
        )
        row, edge_type = row[permutation], edge_type[permutation]
    graph = HomogeneousGraph(
        row=row,
        col=col,
        colptr=offsets,
        edge_type=edge_type,
        num_edge_types=3,
        start_node_offsets={"target": 0},
        end_node_offsets={"target": nodes},
    )
    return replace(
        graph,
        row=graph.row.cuda(),
        col=graph.col.cuda(),
        colptr=graph.colptr.cuda(),
        edge_type=graph.edge_type.cuda(),
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize(
    "case", ["random", "constant", "nonfinite", "empty", "reordered"]
)
@pytest.mark.parametrize("block_size", [1, 64, 128])
def test_cuda_statistics_bitwise(dtype, case, block_size):
    graph = make_cuda_graph(case)
    src = torch.randn(257, 33, device="cuda", dtype=dtype)
    edge_attr = torch.randn(3, 33, device="cuda", dtype=dtype)
    if case == "constant":
        src.fill_(0.5)
        edge_attr.zero_()
    if case == "nonfinite":
        src[3, 0] = float("nan")
        src[10, 1] = float("inf")
        src[17, 2] = -float("inf")
    with torch.inference_mode():
        expected = segment_multi_reduce(
            src, graph.row, edge_attr, graph.colptr, graph.edge_type
        )
        actual = torch.cat(
            [
                destination_statistics(
                    src, graph, edge_attr, start, min(start + block_size, 257)
                )
                for start in range(0, 257, block_size)
            ]
        )
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.parametrize("precision", ["fp32", "bf16", "autocast"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_cuda_module_fit_and_frozen_query(
    precision, block_size, nondefault_stream
):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    graph = make_cuda_graph()
    model = InvariantGNN(32, device="cuda", dtype=dtype).eval()
    candidate = BlockedInvariantGNN(model, block_size)
    x = torch.randn(257, 32, device="cuda", dtype=dtype)
    readout_index = torch.cat(
        [
            torch.arange(257, device="cuda"),
            torch.tensor([256, 0, 128, 64, 0], device="cuda"),
        ]
    )
    kwargs = {
        "graph": graph,
        "readout_table": "target",
        "readout_index": readout_index,
        "num_hops": 2,
    }
    stream = (
        torch.cuda.Stream()
        if nondefault_stream
        else torch.cuda.current_stream()
    )
    stream.wait_stream(torch.cuda.current_stream())
    tolerance = (
        {"rtol": 1e-5, "atol": 2e-6}
        if precision == "fp32"
        else {"rtol": 2e-2, "atol": 2e-2}
    )
    with (
        torch.inference_mode(),
        torch.cuda.stream(stream),
        torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=precision == "autocast"
        ),
    ):
        baseline_cache, candidate_cache = Cache(), Cache()
        expected = model(
            x,
            **kwargs,
            cache=baseline_cache,
            generator=torch.Generator(device="cuda").manual_seed(1729),
        )
        actual = candidate(
            x,
            **kwargs,
            cache=candidate_cache,
            generator=torch.Generator(device="cuda").manual_seed(1729),
        )
        torch.testing.assert_close(actual, expected, **tolerance)
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
    stream.synchronize()
