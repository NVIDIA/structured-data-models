"""Benchmark shared versus member-isolated RFM Recipe execution.

Dataset creation, correctness checks, and host/device transfers are outside the
timed region. CUDA timings use warm-ups, events, and explicit synchronization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import torch

from sdm import EnsembleTable, RelatedTables
from sdm.models.kumorfm.recipe import default_recipe
from sdm.tensor import TableTensor
from sdm.testing.datasets import CanonicalRFMData, Task, canonical_rfm_data

Schedule = Literal["ensemble_shared", "member_isolated"]
BenchmarkOutputs: TypeAlias = (
    tuple[
        EnsembleTable,
        EnsembleTable,
        EnsembleTable,
        tuple[RelatedTables, ...],
        tuple[RelatedTables, ...],
    ]
    | tuple[TableTensor, ...]
)


@dataclass(frozen=True)
class Measurement:
    """One synchronized RFM Recipe measurement."""

    task: Task
    device: str
    schedule: Schedule
    context_rows: int
    query_rows: int
    related_context_rows: dict[str, int]
    related_query_rows: dict[str, int]
    num_estimators: int
    dtype: str
    repetitions: int
    warmups: int
    median_ms: float
    p95_ms: float
    enqueue_median_ms: float | None
    synchronization_wait_median_ms: float | None
    cuda_event_elapsed_median_ms: float | None
    cuda_kernel_time_ms: float | None
    peak_memory_bytes: int
    absolute_peak_memory_bytes: int
    throughput_task_rows_per_second: float


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _cpu_peak(operation: Callable[[], Any]) -> tuple[int, int]:
    baseline = _rss_bytes()
    peak = [baseline]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.001):
            peak[0] = max(peak[0], _rss_bytes())

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        result = operation()
        peak[0] = max(peak[0], _rss_bytes())
        del result
    finally:
        stop.set()
        thread.join()
    return max(0, peak[0] - baseline), peak[0]


def _execute_isolated_member(
    data: CanonicalRFMData,
    *,
    task: Task,
    generator: torch.Generator,
) -> tuple[TableTensor, ...]:
    recipe = default_recipe()
    features, target, related = recipe.fit_transform(
        data.x_context,
        data.target(task),
        data.related_context,
        num_members=1,
        generator=generator,
    )
    query, related_query = recipe.transform(
        data.x_query,
        data.related_query,
    )
    assert related is not None
    assert related_query is not None
    return (
        features.representation(0),
        target.representation(0),
        query.representation(0),
        *related[0].tables.values(),
        *related_query[0].tables.values(),
    )


def _execute(
    data: CanonicalRFMData,
    *,
    task: Task,
    schedule: Schedule,
    num_estimators: int,
) -> BenchmarkOutputs:
    if schedule == "ensemble_shared":
        recipe = default_recipe()
        features, target, related = recipe.fit_transform(
            data.x_context,
            data.target(task),
            data.related_context,
            num_members=num_estimators,
            generator=torch.Generator(
                device=data.x_context.device
            ).manual_seed(42),
        )
        query, related_query = recipe.transform(
            data.x_query,
            data.related_query,
        )
        assert related is not None
        assert related_query is not None
        return features, target, query, related, related_query

    outputs = []
    generator = torch.Generator(device=data.x_context.device).manual_seed(42)
    for _ in range(num_estimators):
        outputs.extend(
            _execute_isolated_member(
                data,
                task=task,
                generator=generator,
            )
        )
    return tuple(outputs)


def _measure(
    data: CanonicalRFMData,
    *,
    task: Task,
    schedule: Schedule,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    profile_kernels: bool,
) -> Measurement:
    device = data.x_context.device

    def operation() -> BenchmarkOutputs:
        with torch.inference_mode():
            return _execute(
                data,
                task=task,
                schedule=schedule,
                num_estimators=num_estimators,
            )

    for _ in range(warmups):
        operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    wall: list[float] = []
    enqueue: list[float] = []
    synchronization_wait: list[float] = []
    cuda_elapsed: list[float] = []
    peak_delta = 0
    absolute_peak = 0
    for _ in range(repetitions):
        start_event: torch.cuda.Event | None = None
        end_event: torch.cuda.Event | None = None
        baseline = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter_ns()
        result = operation()
        if device.type == "cuda":
            assert start_event is not None
            assert end_event is not None
            end_event.record()
            enqueued = time.perf_counter_ns()
            torch.cuda.synchronize(device)
            finished = time.perf_counter_ns()
            enqueue.append((enqueued - started) / 1e6)
            synchronization_wait.append((finished - enqueued) / 1e6)
            cuda_elapsed.append(start_event.elapsed_time(end_event))
            peak = torch.cuda.max_memory_allocated(device)
            peak_delta = max(peak_delta, peak - baseline)
            absolute_peak = max(absolute_peak, peak)
        else:
            finished = time.perf_counter_ns()
        wall.append((finished - started) / 1e6)
        del result

    if device.type == "cpu":
        peak_delta, absolute_peak = _cpu_peak(operation)

    kernel_time: float | None = None
    if device.type == "cuda" and profile_kernels:
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            result = operation()
            torch.cuda.synchronize(device)
        kernel_time = (
            sum(
                event.self_device_time_total
                for event in profiler.key_averages()
            )
            / 1000
        )
        del result

    median = statistics.median(wall)
    cuda_event = statistics.median(cuda_elapsed) if cuda_elapsed else None
    return Measurement(
        task=task,
        device=str(device),
        schedule=schedule,
        context_rows=data.x_context.size(-2),
        query_rows=data.x_query.size(-2),
        related_context_rows={
            name: table.size(-2)
            for name, table in data.related_context.tables.items()
        },
        related_query_rows={
            name: table.size(-2)
            for name, table in data.related_query.tables.items()
        },
        num_estimators=num_estimators,
        dtype=str(data.x_context.numerical.dtype),
        repetitions=repetitions,
        warmups=warmups,
        median_ms=median,
        p95_ms=_p95(wall),
        enqueue_median_ms=(statistics.median(enqueue) if enqueue else None),
        synchronization_wait_median_ms=(
            statistics.median(synchronization_wait)
            if synchronization_wait
            else None
        ),
        cuda_event_elapsed_median_ms=cuda_event,
        cuda_kernel_time_ms=kernel_time,
        peak_memory_bytes=peak_delta,
        absolute_peak_memory_bytes=absolute_peak,
        throughput_task_rows_per_second=(
            (data.x_context.size(-2) + data.x_query.size(-2)) / (median / 1000)
        ),
    )


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _metadata() -> dict[str, object]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the requested canonical RFM Recipe benchmark matrix."""
    measurements = []
    for device_name in args.devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        data = canonical_rfm_data(
            num_context_rows=args.context_rows,
            num_query_rows=args.query_rows,
            num_products=args.products,
            device=device,
        )
        for task in cast(tuple[Task, ...], tuple(args.tasks)):
            for schedule in cast(tuple[Schedule, ...], tuple(args.schedules)):
                measurements.append(
                    _measure(
                        data,
                        task=task,
                        schedule=schedule,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        profile_kernels=args.profile_kernels,
                    )
                )
        del data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "metadata": _metadata(),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "results": [asdict(result) for result in measurements],
    }


def parse_args() -> argparse.Namespace:
    """Parse the benchmark command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=("classification", "regression"),
    )
    parser.add_argument("--devices", nargs="+", default=("cpu", "cuda"))
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=("ensemble_shared", "member_isolated"),
        default=("ensemble_shared", "member_isolated"),
    )
    parser.add_argument("--context-rows", type=int, default=5_000)
    parser.add_argument("--query-rows", type=int, default=1_000)
    parser.add_argument("--products", type=int, default=1_000)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/rfm_ensemble.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and write machine-readable results."""
    args = parse_args()
    output = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    sys.stdout.write(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
