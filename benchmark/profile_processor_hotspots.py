# ruff: noqa: T201

"""Profile the representative GPU Processor hotspots with Nsight Systems."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmark.processor_characterization import (
    DEFAULTS,
    SWEEPS,
    Sweep,
    _prepare,
)


@dataclass(frozen=True)
class Scenario:
    """One fixed Processor hotspot scenario."""

    processor: str
    operation: str
    axis: str
    parameters: dict[str, Any]


SCENARIOS = {
    "shuffle_categories": Scenario(
        processor="ShuffleCategories",
        operation="fit_transform_ensemble",
        axis="categorical_columns",
        parameters={
            "rows": 10_000,
            "categorical_columns": 32,
            "cardinality": 1_024,
            "estimators": 8,
        },
    ),
    "power_transform": Scenario(
        processor="PowerTransform",
        operation="fit_transform",
        axis="rows",
        parameters={
            "rows": 50_000,
            "columns": 32,
            "dtype": "float32",
        },
    ),
    "align_categories": Scenario(
        processor="AlignCategories",
        operation="fit_transform",
        axis="categorical_columns",
        parameters={
            "rows": 10_000,
            "categorical_columns": 32,
            "cardinality": 1_024,
        },
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sweep(scenario: Scenario) -> Sweep:
    return next(
        sweep
        for sweep in SWEEPS
        if sweep.processor == scenario.processor
        and sweep.operation == scenario.operation
        and sweep.axis.name == scenario.axis
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def main() -> None:
    """Run one hotspot inside a filterable NVTX range."""
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    scenario = SCENARIOS[args.scenario]
    params = dict(DEFAULTS)
    params.update(scenario.parameters)
    device = torch.device("cuda")
    prepare = _prepare(_sweep(scenario), params, device)

    for _ in range(args.warmups):
        result = prepare()()
        torch.cuda.synchronize(device)
        del result

    runs = [prepare() for _ in range(args.iterations)]
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)

    wall_ms: list[float] = []
    event_ms: list[float] = []
    torch.cuda.nvtx.range_push(args.scenario)
    try:
        for index, run in enumerate(runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"iteration_{index}")
            start_event.record()
            started = time.perf_counter()
            result = run()
            end_event.record()
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            torch.cuda.nvtx.range_pop()
            wall_ms.append((finished - started) * 1_000)
            event_ms.append(start_event.elapsed_time(end_event))
            del result
    finally:
        torch.cuda.nvtx.range_pop()

    gpu = torch.cuda.get_device_properties(device)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario": args.scenario,
        "processor": scenario.processor,
        "operation": scenario.operation,
        "parameters": scenario.parameters,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "hardware": {
            "gpu": gpu.name,
            "total_memory_bytes": gpu.total_memory,
            "compute_capability": f"{gpu.major}.{gpu.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "measurement": {
            "wall_median_ms": statistics.median(wall_ms),
            "wall_p95_ms": _percentile(wall_ms, 0.95),
            "cuda_event_median_ms": statistics.median(event_ms),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_delta_bytes": (
                torch.cuda.max_memory_allocated(device) - baseline_bytes
            ),
            "wall_samples_ms": wall_ms,
            "cuda_event_samples_ms": event_ms,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
