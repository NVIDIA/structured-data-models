# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import (
    DAG,
    LogUniformInt,
    MixtureDAGSampler,
    RandomCauchyDAGSampler,
)


def _graph(
    input_nodes: torch.Tensor | None = None,
    target_node: int | None = None,
) -> DAG:
    # 0 -> 1, {0, 1} -> 2, and the disconnected node 3.
    return DAG(
        parent_ptr=torch.tensor([0, 0, 1, 3, 3]),
        parent_idx=torch.tensor([0, 0, 1]),
        input_nodes=input_nodes,
        target_node=target_node,
    )


def test_graph_traversal() -> None:
    graph = _graph(torch.tensor([1]), 2)
    assert graph.num_nodes == 4
    assert graph.num_edges == 3
    assert [graph.parents(i).tolist() for i in range(4)] == [
        [],
        [0],
        [0, 1],
        [],
    ]
    assert graph.topological_order.tolist() == [0, 1, 2, 3]
    assert graph.pruned_topological_order.tolist() == [0, 1, 2]
    assert _graph().pruned_topological_order.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("input_node", "target_node", "expected"),
    [(1, 2, True), (2, 0, True), (3, 2, False), (2, 2, True)],
)
def test_graph_connectivity(
    input_node: int, target_node: int, expected: bool
) -> None:
    assert (
        _graph(torch.tensor([input_node]), target_node).is_valid() is expected
    )


@pytest.mark.parametrize("num_nodes", [1, 6])
def test_cauchy_sampling(num_nodes: int) -> None:
    graph = RandomCauchyDAGSampler(num_nodes).sample()
    assert graph.num_nodes == num_nodes
    assert graph.input_nodes is None
    assert graph.target_node is None
    assert graph.parent_ptr.dtype == graph.parent_idx.dtype == torch.int64
    assert int(graph.parent_ptr[0]) == 0
    assert int(graph.parent_ptr[-1]) == graph.num_edges
    for node in range(num_nodes):
        assert all(
            0 <= parent < node for parent in graph.parents(node).tolist()
        )


def test_column_assignment_replay_and_cpu_sampling() -> None:
    sampler = MixtureDAGSampler([RandomCauchyDAGSampler(LogUniformInt(2, 32))])
    num_columns = 3
    generator = torch.Generator()
    state = generator.get_state()
    global_state = torch.get_rng_state()
    with torch.device("meta"):
        graph = sampler.sample(generator)
        assigned = graph.with_columns(num_columns, generator)
    final_state = generator.get_state()

    assert graph.input_nodes is None
    assert graph.target_node is None
    assert assigned.input_nodes is not None
    assert assigned.input_nodes.shape == (num_columns,)
    assert all(
        0 <= node < graph.num_nodes for node in assigned.input_nodes.tolist()
    )
    assert assigned.target_node is not None
    assert 0 <= assigned.target_node < graph.num_nodes
    for tensor in (
        assigned.parent_ptr,
        assigned.parent_idx,
        assigned.input_nodes,
    ):
        assert tensor.device.type == "cpu"
    torch.testing.assert_close(assigned.parent_ptr, graph.parent_ptr)
    torch.testing.assert_close(assigned.parent_idx, graph.parent_idx)

    generator.set_state(state)
    replay = sampler.sample(generator).with_columns(num_columns, generator)
    for name in ("parent_ptr", "parent_idx", "input_nodes", "target_node"):
        torch.testing.assert_close(
            getattr(assigned, name), getattr(replay, name)
        )
    assert torch.equal(generator.get_state(), final_state)
    assert not torch.equal(state, final_state)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_mixture_uses_weights() -> None:
    sampler = MixtureDAGSampler(
        [(RandomCauchyDAGSampler(1), 0.0), RandomCauchyDAGSampler(6)]
    )
    assert sampler.sample().num_nodes == 6
