# ruff: noqa: D103, TID253
"""Independent adversarial and optional real-checkpoint GNN checks."""

import hashlib
import inspect
import json
import os
from functools import cache
from pathlib import Path

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


@cache
def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_from_degrees(degrees: torch.Tensor) -> HomogeneousGraph:
    nodes = len(degrees)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), degrees.cumsum(0)])
    edges = int(offsets[-1])
    return HomogeneousGraph(
        row=torch.arange(edges).mul(7).remainder(nodes),
        col=torch.arange(nodes).repeat_interleave(degrees),
        colptr=offsets,
        edge_type=torch.arange(edges).remainder(3),
        num_edge_types=3,
        start_node_offsets={"other": 0, "readout": 2},
        end_node_offsets={"other": 2, "readout": nodes},
    )


def test_variance_threshold_and_empty_singleton_destinations() -> None:
    graph = graph_from_degrees(torch.tensor([0, 2, 1, 2, 0]))
    graph.row[:] = torch.tensor([0, 1, 2, 0, 1])
    src = torch.tensor(
        [[-0.002, -0.004], [0.002, 0.004], [1.0, 1.0], [0, 0], [0, 0]]
    )
    edge_attr = torch.zeros(3, 2)
    expected = segment_multi_reduce(
        src, graph.row, edge_attr, graph.colptr, graph.edge_type
    )
    for block_size in [1, 2, 4, 8]:
        actual = torch.cat(
            [
                destination_statistics(
                    src, graph, edge_attr, start, min(start + block_size, 5)
                )
                for start in range(0, 5, block_size)
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[:, 2],
            torch.tensor([[0, 0], [0, 0.004], [0, 0], [0, 0.004], [0, 0]]),
        )


@pytest.mark.parametrize("checkpoint", ["classifier", "regressor"])
@pytest.mark.parametrize("precision", ["fp32", "bf16", "autocast_bf16"])
@pytest.mark.parametrize(
    ("nodes", "block_size"),
    [(64, 1), (64, 17), (64, 256), (257, 64), (257, 128)],
)
def test_actual_checkpoint_gnn_and_frozen_cache(
    checkpoint: str, precision: str, nodes: int, block_size: int
) -> None:
    root = os.environ.get("SDM_RELATIONAL_CHECKPOINT_ROOT")
    if root is None:
        pytest.skip("Set SDM_RELATIONAL_CHECKPOINT_ROOT for offline weights")
    state = torch.load(
        Path(root) / f"{checkpoint}.pt", map_location="cpu", weights_only=True
    )
    gnn_state = {
        key.removeprefix("gnn."): value
        for key, value in state.items()
        if key.startswith("gnn.")
    }
    model = InvariantGNN(512).eval()
    model.load_state_dict(gnn_state)
    if precision == "bf16":
        model.bfloat16()
    candidate = BlockedInvariantGNN(model, block_size=block_size)
    degrees = torch.tensor([0, 1, 3, 17]).repeat((nodes + 3) // 4)[:nodes]
    graph = graph_from_degrees(degrees)
    x = torch.arange(nodes * 512).reshape(nodes, 512).float().mul(0.01).sin()
    if precision == "bf16":
        x = x.bfloat16()
    kwargs = {
        "graph": graph,
        "readout_table": "readout",
        "readout_index": torch.tensor([nodes - 3, 0, 13, 0, 39]),
        "num_hops": 2,
    }
    tolerance = (
        {"rtol": 1e-5, "atol": 2e-6}
        if precision == "fp32"
        else {"rtol": 2e-2, "atol": 2e-2}
    )
    baseline_cache, candidate_cache = Cache(), Cache()
    with (
        torch.inference_mode(),
        torch.autocast(
            "cpu", dtype=torch.bfloat16, enabled=precision == "autocast_bf16"
        ),
    ):
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
        comparisons = [("fit", actual.clone(), expected.clone())]
        torch.testing.assert_close(
            candidate_cache["edge_type_emb"],
            baseline_cache["edge_type_emb"],
            rtol=0,
            atol=0,
        )
        baseline_cache.freeze()
        candidate_cache.freeze()
        expected = model(x.flip(0), **kwargs, cache=baseline_cache)
        actual = candidate(x.flip(0), **kwargs, cache=candidate_cache)
        comparisons.append(("frozen_query", actual, expected))
    record = {
        "evidence_kind": "CPU GNN-only, synthetic inputs, actual weights",
        "torch_version": torch.__version__,
        "checkpoint": checkpoint,
        "checkpoint_revision": "2bd603d3d8f25f67a7aaa8579908595e20567e22",
        "checkpoint_sha256": file_sha256(str(Path(root) / f"{checkpoint}.pt")),
        "adapter_source_sha256": file_sha256(
            inspect.getfile(BlockedInvariantGNN)
        ),
        "test_source_sha256": file_sha256(__file__),
        "nodes": nodes,
        "edges": graph.num_edges,
        "channels": 512,
        "degree_pattern": [0, 1, 3, 17],
        "feature_formula": "sin(arange(nodes*512)*0.01)",
        "edge_rng_seed": 1729,
        "num_hops": 2,
        "block_size": block_size,
        "precision": precision,
        "tolerance": tolerance,
        "comparisons": {},
    }
    for phase, actual, expected in comparisons:
        difference = (actual.float() - expected.float()).abs()
        accepted = difference <= (
            tolerance["atol"] + tolerance["rtol"] * expected.float().abs()
        )
        record["comparisons"][phase] = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "mismatched_elements": int((~accepted).sum()),
            "elements": actual.numel(),
            "pass": bool(accepted.all()),
            "bitwise_equal": bool(torch.equal(actual, expected)),
        }
    output = os.environ.get("SDM_GNN_REVIEW_OUTPUT")
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        name = f"{checkpoint}-{precision}-n{nodes}-b{block_size}.json"
        (destination / name).write_text(json.dumps(record, indent=2) + "\n")
    for _, actual, expected in comparisons:
        torch.testing.assert_close(actual, expected, **tolerance)
