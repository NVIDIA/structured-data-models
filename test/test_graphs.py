# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch

from sdm._graphs import GraphCache
from sdm.testing import onlyCUDA


@onlyCUDA
def test_graph_cache_replays_repeated_calls() -> None:
    linear = torch.nn.Linear(8, 4, device="cuda")
    graphs = GraphCache()

    with torch.inference_mode():
        for _ in range(4):
            x = torch.randn(5, 8, device="cuda")
            torch.testing.assert_close(graphs(linear, x, key="a"), linear(x))

        # In-place weight updates are seen by replays:
        linear.weight.mul_(2)
        x = torch.randn(5, 8, device="cuda")
        torch.testing.assert_close(graphs(linear, x, key="a"), linear(x))

        # Other shapes are separate entries:
        x = torch.randn(3, 8, device="cuda")
        torch.testing.assert_close(graphs(linear, x, key="a"), linear(x))


@onlyCUDA
def test_graph_cache_outputs_do_not_alias() -> None:
    graphs = GraphCache()
    with torch.inference_mode():
        outs = [
            graphs(torch.exp, torch.full((4,), float(i), device="cuda"), key=0)
            for i in range(3)
        ]
    for i, out in enumerate(outs):
        torch.testing.assert_close(out, torch.full_like(out, i).exp())


@onlyCUDA
def test_graph_cache_autocast() -> None:
    linear = torch.nn.Linear(8, 4, device="cuda")
    graphs = GraphCache()
    x = torch.randn(5, 8, device="cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected = linear(x)
        for _ in range(3):
            out = graphs(linear, x, key="a")

    assert out.dtype == torch.float16
    torch.testing.assert_close(out, expected)


@onlyCUDA
def test_graph_cache_replays_launch_bound_calls() -> None:
    graphs = GraphCache()
    num_runs = 0

    def fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal num_runs
        num_runs += 1
        for _ in range(200):  # Many tiny kernels wait on their launches.
            x = x + 1
        return x

    with torch.inference_mode():
        for _ in range(10):
            x = torch.randn(4, device="cuda")
            torch.testing.assert_close(graphs(fn, x, key="a"), x + 200)

    # Replays run the captured kernels without running the Python function:
    assert num_runs < 10


@onlyCUDA
def test_graph_cache_runs_eagerly_when_replays_are_not_faster() -> None:
    graphs = GraphCache()
    num_runs = 0

    def fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal num_runs
        num_runs += 1
        return x * 2

    with torch.inference_mode():
        x = torch.randn(2**24, device="cuda")
        for _ in range(10):
            torch.testing.assert_close(graphs(fn, x, key="a"), x * 2)
        num_runs = 0
        for _ in range(5):
            torch.testing.assert_close(graphs(fn, x, key="a"), x * 2)

    # A single large kernel does not wait on its launch, while a replay also
    # copies the input and output, so the graph is dropped and calls run the
    # function again:
    assert num_runs == 5


def test_graph_cache_copy_starts_empty() -> None:
    graphs = GraphCache()
    x = torch.zeros(2)
    graphs(torch.exp, x, key=0)

    # A copy has not seen the call yet and runs it eagerly instead of
    # capturing a graph:
    copied = copy.deepcopy(graphs)
    torch.testing.assert_close(copied(torch.exp, x, key=0), x.exp())
