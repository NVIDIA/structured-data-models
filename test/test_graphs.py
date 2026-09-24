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


def test_graph_cache_copy_starts_empty() -> None:
    graphs = GraphCache()
    x = torch.zeros(2)
    graphs(torch.exp, x, key=0)

    # A copy has not seen the call yet and runs it eagerly instead of
    # capturing a graph:
    copied = copy.deepcopy(graphs)
    torch.testing.assert_close(copied(torch.exp, x, key=0), x.exp())
