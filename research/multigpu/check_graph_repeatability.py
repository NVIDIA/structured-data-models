# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check edge ordering and graph identity across repeated CUDA graph builds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdm.models.kumo.relational.task import TaskGraph


def check(workload: Path, output: Path, repeats: int) -> None:
    """Inspect one fixed sampled query batch; never load weights or targets."""
    payload = torch.load(workload / "graphs.pt", weights_only=False)
    sample = payload["queries"][0]
    x = sample.task_table.cuda()
    related = sample.related_tables.to("cuda:0")
    reference = None
    records = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        task = TaskGraph.from_input(x, related, num_hops=2)
        torch.cuda.synchronize()
        edges = (
            torch.stack(
                [task.graph.col, task.graph.row, task.graph.edge_type], dim=1
            )
            .cpu()
            .numpy()
        )
        canonical = edges[np.lexsort(edges.T[::-1])]
        assignments = {
            name: rows.cpu().numpy()
            for name, rows in task.task_row_by_table.items()
        }
        current = (edges, canonical, assignments, task.readout_index.cpu())
        if reference is None:
            reference = current
        records.append(
            {
                "edge_order_equal": bool(np.array_equal(edges, reference[0])),
                "edge_multiset_equal": bool(
                    np.array_equal(canonical, reference[1])
                ),
                "edge_positions_changed": int(
                    np.any(edges != reference[0], axis=1).sum()
                ),
                "task_assignments_equal": all(
                    np.array_equal(value, reference[2][name])
                    for name, value in assignments.items()
                ),
                "readout_equal": bool(torch.equal(current[3], reference[3])),
                "directed_edges": len(edges),
            }
        )
    result = {
        "workload": str(workload),
        "query_batch": 0,
        "synchronization": "Before and after each full TaskGraph construction",
        "repeats": records,
        "scope": "Graph diagnostic only, no timing or prediction claim",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    check(args.workload, args.output, args.repeats)
