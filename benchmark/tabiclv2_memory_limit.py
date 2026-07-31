"""Find the TabICLv2 parallel row limit on the available CUDA device."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from benchmark.tabiclv2_ensemble import build_workload, metadata
from sdm.models import TabICLv2

Mode = Literal["parallel", "sequential"]


@dataclass(frozen=True)
class Attempt:
    """Outcome of one isolated row-count attempt."""

    task: str
    rows: int
    context_rows: int
    query_rows: int
    mode: Mode
    success: bool
    runtime_ms: float | None
    baseline_memory_bytes: int
    peak_memory_bytes: int
    peak_reserved_memory_bytes: int


def attempt(
    model: TabICLv2,
    *,
    task: str,
    rows: int,
    features: int,
    categorical_features: int,
    vocabulary_size: int,
    num_estimators: int,
    mode: Mode,
) -> Attempt:
    """Run one clean attempt and recover from CUDA OOM."""
    context_rows = max(2, int(rows * 0.8))
    query_rows = rows - context_rows
    if query_rows < 1:
        raise ValueError("Expected at least one query row.")

    gc.collect()
    torch.cuda.empty_cache()
    workload = build_workload(
        task=task,
        context_rows=context_rows,
        query_rows=query_rows,
        features=features,
        categorical_features=categorical_features,
        vocabulary_size=vocabulary_size,
    ).to("cuda")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = None
    started = time.perf_counter_ns()
    success = False
    runtime_ms: float | None = None
    try:
        output = model(
            workload.x_context,
            workload.target,
            workload.x_query,
            num_estimators=num_estimators,
            ensemble_mode=mode,
            generator=torch.Generator().manual_seed(42),
        )
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter_ns() - started) / 1e6
        success = True
    except torch.cuda.OutOfMemoryError:
        pass
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del output, workload
    gc.collect()
    torch.cuda.empty_cache()
    return Attempt(
        task=task,
        rows=rows,
        context_rows=context_rows,
        query_rows=query_rows,
        mode=mode,
        success=success,
        runtime_ms=runtime_ms,
        baseline_memory_bytes=baseline,
        peak_memory_bytes=peak,
        peak_reserved_memory_bytes=peak_reserved,
    )


def _rounded_midpoint(low: int, high: int, resolution: int) -> int:
    midpoint = (low + high) // 2
    rounded = midpoint // resolution * resolution
    return min(high - resolution, max(low + resolution, rounded))


def search(args: argparse.Namespace) -> dict[str, object]:
    """Run exponential growth followed by a resolution-bounded search."""
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA device is required.")
    if args.resolution < 1:
        raise ValueError("'resolution' must be positive.")

    torch.manual_seed(0)
    model = TabICLv2(pretrained=False, device="cuda").eval()
    attempts: list[Attempt] = []
    successful_rows = 0
    failing_rows: int | None = None

    candidate = args.start_rows
    while True:
        result = attempt(
            model,
            task=args.task,
            rows=candidate,
            features=args.features,
            categorical_features=args.categorical_features,
            vocabulary_size=args.vocabulary_size,
            num_estimators=args.num_estimators,
            mode="parallel",
        )
        attempts.append(result)
        if not result.success:
            failing_rows = candidate
            break
        successful_rows = candidate
        if candidate >= args.max_rows:
            break
        candidate = min(args.max_rows, candidate * 2)

    if failing_rows is not None:
        low = successful_rows
        high = failing_rows
        while high - low > args.resolution:
            candidate = _rounded_midpoint(low, high, args.resolution)
            result = attempt(
                model,
                task=args.task,
                rows=candidate,
                features=args.features,
                categorical_features=args.categorical_features,
                vocabulary_size=args.vocabulary_size,
                num_estimators=args.num_estimators,
                mode="parallel",
            )
            attempts.append(result)
            if result.success:
                low = candidate
            else:
                high = candidate
        successful_rows = low
        failing_rows = high

    sequential_at_failure: Attempt | None = None
    if failing_rows is not None:
        sequential_at_failure = attempt(
            model,
            task=args.task,
            rows=failing_rows,
            features=args.features,
            categorical_features=args.categorical_features,
            vocabulary_size=args.vocabulary_size,
            num_estimators=args.num_estimators,
            mode="sequential",
        )
        attempts.append(sequential_at_failure)

    largest = next(
        (
            result
            for result in reversed(attempts)
            if result.mode == "parallel"
            and result.success
            and result.rows == successful_rows
        ),
        None,
    )
    first_failure = next(
        (
            result
            for result in reversed(attempts)
            if result.mode == "parallel"
            and not result.success
            and result.rows == failing_rows
        ),
        None,
    )
    return {
        "metadata": metadata(),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "largest_success": asdict(largest) if largest is not None else None,
        "first_failure": (
            asdict(first_failure) if first_failure is not None else None
        ),
        "sequential_at_first_failure": (
            asdict(sequential_at_failure)
            if sequential_at_failure is not None
            else None
        ),
        "attempts": [asdict(result) for result in attempts],
    }


def parse_args() -> argparse.Namespace:
    """Parse the memory-search CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--start-rows", type=int, default=2_000)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--resolution", type=int, default=1_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--categorical-features", type=int, default=10)
    parser.add_argument("--vocabulary-size", type=int, default=4_096)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/tabiclv2_l4_memory_limit.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the search and persist its complete attempt history."""
    args = parse_args()
    result = search(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
