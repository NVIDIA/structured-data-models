"""Find the TabICLv2 ensemble preprocessing row limit on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmark.tabiclv2_ensemble import Task, build_workload, metadata
from sdm.models.tabiclv2.recipe import default_recipe


@dataclass(frozen=True)
class Attempt:
    """Outcome of one isolated preprocessing attempt."""

    task: Task
    rows: int
    context_rows: int
    query_rows: int
    success: bool
    failed_stage: str | None
    runtime_ms: float | None
    baseline_memory_bytes: int
    peak_delta_memory_bytes: int
    peak_memory_bytes: int
    peak_reserved_memory_bytes: int


def attempt(
    *,
    task: Task,
    rows: int,
    features: int,
    categorical_features: int,
    vocabulary_size: int,
    num_estimators: int,
) -> Attempt:
    """Run one clean Recipe attempt and recover from CUDA OOM."""
    context_rows = max(2, int(rows * 0.8))
    query_rows = rows - context_rows
    if query_rows < 1:
        raise ValueError("Expected at least one query row.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cpu_workload = None
    workload = None
    recipe = None
    context = None
    target = None
    query = None
    baseline = 0
    runtime_ms: float | None = None
    failed_stage: str | None = "input_transfer"
    success = False
    try:
        cpu_workload = build_workload(
            task=task,
            context_rows=context_rows,
            query_rows=query_rows,
            features=features,
            categorical_features=categorical_features,
            vocabulary_size=vocabulary_size,
        )
        workload = cpu_workload.to("cuda")
        cpu_workload = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        recipe = default_recipe()
        started = time.perf_counter_ns()
        failed_stage = "fit_transform"
        with torch.inference_mode():
            context, target, _ = recipe.fit_transform(
                workload.x_context,
                workload.target,
                num_members=num_estimators,
                generator=torch.Generator().manual_seed(42),
            )
            failed_stage = "query_transform"
            query, _ = recipe.transform(workload.x_query)
        torch.cuda.synchronize()
        runtime_ms = (time.perf_counter_ns() - started) / 1e6
        if not (
            context.num_members
            == target.num_members
            == query.num_members
            == num_estimators
        ):
            raise RuntimeError(
                "Expected one preprocessing result per ensemble member."
            )
        failed_stage = None
        success = True
    except torch.cuda.OutOfMemoryError:
        pass

    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del context, target, query, recipe, workload, cpu_workload
    gc.collect()
    torch.cuda.empty_cache()
    return Attempt(
        task=task,
        rows=rows,
        context_rows=context_rows,
        query_rows=query_rows,
        success=success,
        failed_stage=failed_stage,
        runtime_ms=runtime_ms,
        baseline_memory_bytes=baseline,
        peak_delta_memory_bytes=max(0, peak - baseline),
        peak_memory_bytes=peak,
        peak_reserved_memory_bytes=peak_reserved,
    )


def search(args: argparse.Namespace) -> dict[str, object]:
    """Run exponential growth followed by a resolution-bounded search."""
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA device is required.")
    if args.resolution < 1:
        raise ValueError("'resolution' must be positive.")
    if not 1 <= args.start_rows <= args.max_rows:
        raise ValueError("Expected 'start_rows' between one and 'max_rows'.")

    attempts: list[Attempt] = []
    successful_rows = 0
    failing_rows: int | None = None
    candidate = args.start_rows
    while True:
        result = attempt(
            task=args.task,
            rows=candidate,
            features=args.features,
            categorical_features=args.categorical_features,
            vocabulary_size=args.vocabulary_size,
            num_estimators=args.num_estimators,
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
            midpoint = (low + high) // 2
            candidate = min(
                high - args.resolution,
                max(
                    low + args.resolution,
                    midpoint // args.resolution * args.resolution,
                ),
            )
            result = attempt(
                task=args.task,
                rows=candidate,
                features=args.features,
                categorical_features=args.categorical_features,
                vocabulary_size=args.vocabulary_size,
                num_estimators=args.num_estimators,
            )
            attempts.append(result)
            if result.success:
                low = candidate
            else:
                high = candidate
        successful_rows = low
        failing_rows = high

    largest = max(
        (result for result in attempts if result.success),
        key=lambda result: result.rows,
        default=None,
    )
    first_failure = min(
        (result for result in attempts if not result.success),
        key=lambda result: result.rows,
        default=None,
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
        "attempts": [asdict(result) for result in attempts],
    }


def parse_args() -> argparse.Namespace:
    """Parse the preprocessing memory-search CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--start-rows", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--resolution", type=int, default=50_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--categorical-features", type=int, default=10)
    parser.add_argument("--vocabulary-size", type=int, default=4_096)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark/results/tabiclv2_preprocessing_memory_limit.json"
        ),
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
