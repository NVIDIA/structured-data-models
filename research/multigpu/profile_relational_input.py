# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure CPU graph processing and estimate GPU activation sizes."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

import sdm
from sdm.models.kumo.relational.task import TaskGraph
from sdm.processing.execution import RecipeExecution


def profile(workload: Path, estimators: int, output: Path) -> None:
    """Profile prepared native input without model execution or GPU work."""
    payload = torch.load(workload / "graphs.pt", weights_only=False)
    metadata = json.loads((workload / "workload.json").read_text())
    sample = payload["context"]
    target = metadata["target"]
    start = time.perf_counter()
    graph = TaskGraph.from_input(
        sample.task_table.drop_columns(target),
        sample.related_tables,
        num_hops=2,
    )
    graph_s = time.perf_counter() - start
    graph_warm_s = []
    for _ in range(3):
        start = time.perf_counter()
        graph = TaskGraph.from_input(
            sample.task_table.drop_columns(target),
            sample.related_tables,
            num_hops=2,
        )
        graph_warm_s.append(time.perf_counter() - start)
    execution = RecipeExecution(sdm.models.KumoRelational.default_recipe())
    start = time.perf_counter()
    with torch.inference_mode():
        members = execution.fit_transform(
            x=sample.task_table.drop_columns(target),
            y=sample.task_table[target],
            related_tables=sample.related_tables,
            num_members=estimators,
            generator=torch.Generator().manual_seed(20261008),
        )
    preprocessing_s = time.perf_counter() - start
    tables = {}
    for name, table in members[0].related_tables.tables.items():
        rows, columns = table.numerical.shape
        # Relative-time features are injected by the model after the recipe.
        relative_times = table.datetime.size(-1) * members[0].x.datetime.size(
            -1
        )
        injected_task = (
            members[0].x.numerical.size(-1)
            if name == graph.readout_table
            else 0
        )
        width = columns + relative_times + injected_task
        tables[name] = {
            "rows": rows,
            "numeric_features_after_recipe": columns,
            "numeric_dtype": str(table.numerical.dtype),
            "features_with_model_injections": width,
            "bf16_row_embedding_buffer_bytes": rows * (4 + width) * 128 * 2,
        }
    nodes = graph.graph.num_nodes
    result = {
        "workload": str(workload),
        "device": "cpu",
        "torch": torch.__version__,
        "estimators": estimators,
        "context": metadata["context"],
        "task_graph_seconds": graph_s,
        "task_graph_warm_seconds": graph_warm_s,
        "recipe_seconds": preprocessing_s,
        "graph_nodes": nodes,
        "graph_directed_edges": graph.graph.num_edges,
        "graph_index_bytes": sum(
            t.numel() * t.element_size()
            for t in [
                graph.graph.row,
                graph.graph.col,
                graph.graph.colptr,
                graph.graph.edge_type,
            ]
        ),
        "tables": tables,
        "estimated_per_member_bf16_gnn_node_embedding_bytes": nodes * 512 * 2,
        "estimated_per_member_bf16_gnn_statistics_bytes": nodes * 5 * 512 * 2,
        "estimated_per_member_bf16_icl_kv_bytes": metadata["context"]
        * 12
        * 2
        * 512
        * 2,
        "estimate_caveat": (
            "Tensor-shape arithmetic only, not measured GPU peaks; "
            "excludes other live buffers, allocator and row-column caches"
        ),
        "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.workload, args.estimators, args.output)
