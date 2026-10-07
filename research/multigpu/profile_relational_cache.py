# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect actual fitted CPU KumoRelational cache tensors on real input."""

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch

import sdm


def profile(workload: Path, output: Path) -> None:
    """Fit one member on CPU and record nested cache tensor sizes."""
    metadata = json.loads((workload / "workload.json").read_text())
    sample = torch.load(workload / "graphs.pt", weights_only=False)["context"]
    model = sdm.models.KumoRelational(task=metadata["problem"], device="cpu")
    events = []
    started = {}

    def before(module: torch.nn.Module, args: tuple) -> None:
        started[id(module)] = time.perf_counter()

    def after(module: torch.nn.Module, args: tuple, result: object) -> None:
        events.append(
            {
                "module": type(module).__name__,
                "seconds": time.perf_counter() - started.pop(id(module)),
            }
        )

    handles = []
    inner = next(iter(model.models.values()))
    for module in [inner.row_embedding, inner.gnn, inner.icl_block]:
        handles.extend(
            [
                module.register_forward_pre_hook(before),
                module.register_forward_hook(after),
            ]
        )
    start = time.perf_counter()
    model.fit(
        x=sample.task_table.drop_columns(metadata["target"]),
        y=sample.task_table[metadata["target"]],
        related_tables=sample.related_tables,
        num_estimators=1,
        generator=torch.Generator().manual_seed(20261008),
    )
    fit_seconds = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    tensors = []
    storages = {}

    def visit(value: object, name: str) -> None:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = storage.nbytes()
            tensors.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                    "retained_storage_bytes": storage.nbytes(),
                }
            )
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{name}/{key}")
        elif isinstance(value, (tuple, list)):
            for key, item in enumerate(value):
                visit(item, f"{name}/{key}")

    visit(model._cache, "cache")
    result = {
        "workload": str(workload),
        "device": "cpu",
        "precision": "float32",
        "context": metadata["context"],
        "fit_seconds": fit_seconds,
        "module_events": events,
        "enumerated_cache_tensor_bytes": sum(t["bytes"] for t in tensors),
        "enumerated_cache_unique_storage_bytes": sum(storages.values()),
        "tensors": tensors,
        "caveat": (
            "CPU inspection only, not a GPU phase profile. "
            "Tensor/storage bytes exclude processor/module buffers."
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(  # noqa: T201
        json.dumps({k: v for k, v in result.items() if k != "tensors"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile(args.workload, args.output)
