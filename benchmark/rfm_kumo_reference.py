"""Benchmark the exact kumo-ml RFM reference on SDM's canonical dataset.

Run this module from an environment containing kumo-ml commit
``e978685d46478758a09e3e472ea2fde9b5c14957`` and ``kumo-api==0.92.0``.
Dataset construction and model loading are outside the measured intervals.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # noqa: TID253
import torch
from kumoapi.model_plan import RunMode
from kumoapi.rfm import InferenceConfig
from kumoapi.rfm.context import (
    REV_REL,
    Context,
    EdgeLayout,
    Link,
    Subgraph,
    Table,
)
from kumoapi.task import TaskType
from kumoapi.typing import Stype as KumoStype
from kumoml.common._version import git_version
from kumoml.rfm.base.data import Data
from kumoml.rfm.v2_1.driver import KumoRFMDriverV2_1

from benchmark.rfm_ensemble import _cpu_peak, _metadata, _p95
from sdm.testing.datasets import Task, canonical_rfm_data

Stage = Literal["data_from_context", "predict"]
REFERENCE_COMMIT = "e978685d46478758a09e3e472ea2fde9b5c14957"


@dataclass(frozen=True)
class ReferenceMeasurement:
    """One synchronized kumo-ml reference measurement."""

    task: Task
    device: str
    stage: Stage
    context_rows: int
    query_rows: int
    products: int
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


def build_context(
    task: Task,
    *,
    context_rows: int,
    query_rows: int,
    products: int,
) -> Context:
    """Convert SDM's canonical relational dataset to a kumo-api Context."""
    data = canonical_rfm_data(
        num_context_rows=context_rows,
        num_query_rows=query_rows,
        num_products=products,
    )
    total_rows = context_rows + query_rows
    batches = np.arange(total_rows, dtype=np.int64)
    order_batches = np.repeat(batches, 2)

    customers = pd.concat(
        [
            data.related_context.tables["customers"].to_pandas(),
            data.related_query.tables["customers"].to_pandas(),
        ],
        ignore_index=True,
    )
    orders = pd.concat(
        [
            data.related_context.tables["orders"].to_pandas(),
            data.related_query.tables["orders"].to_pandas(),
        ],
        ignore_index=True,
    )
    product_frames = [
        data.related_context.tables["products"].to_pandas(),
        data.related_query.tables["products"].to_pandas(),
    ]
    product_rows = orders["product_id"].to_numpy(dtype=np.int64) - 10_000
    product_rows[order_batches >= context_rows] += products
    task_frame = pd.concat(
        [data.x_context.to_pandas(), data.x_query.to_pandas()],
        ignore_index=True,
    )

    table_dict = {
        "customers": Table(
            df=customers,
            row=None,
            batch=batches,
            num_sampled_nodes=[total_rows, 0, 0],
            stype_dict={
                "age": KumoStype.numerical,
                "lifetime_spend": KumoStype.numerical,
                "segment": KumoStype.categorical,
                "signup_at": KumoStype.timestamp,
                "constant_customer": KumoStype.numerical,
            },
            primary_key="customer_id",
        ),
        "orders": Table(
            df=orders,
            row=None,
            batch=order_batches,
            num_sampled_nodes=[0, len(orders), 0],
            stype_dict={
                "amount": KumoStype.numerical,
                "channel": KumoStype.categorical,
                "ordered_at": KumoStype.timestamp,
                "constant_order": KumoStype.numerical,
            },
            primary_key="order_id",
        ),
        "products": Table(
            df=pd.concat(product_frames, ignore_index=True),
            row=product_rows,
            batch=order_batches,
            num_sampled_nodes=[0, 0, len(product_rows)],
            stype_dict={
                "category": KumoStype.categorical,
                "price": KumoStype.numerical,
                "released_at": KumoStype.timestamp,
                "constant_product": KumoStype.numerical,
            },
            primary_key=None,
        ),
    }
    num_orders = len(orders)
    link_dict = {
        ("orders", "customer_id", "customers"): Link(
            layout=EdgeLayout.CSC,
            row=None,
            col=np.arange(0, num_orders + 1, 2, dtype=np.int64),
            num_sampled_edges=[num_orders, 0],
        ),
        ("customers", f"{REV_REL}customer_id", "orders"): Link(
            layout=EdgeLayout.REV,
            row=None,
            col=None,
            num_sampled_edges=[0, num_orders],
        ),
        ("products", f"{REV_REL}product_id", "orders"): Link(
            layout=EdgeLayout.COO,
            row=None,
            col=None,
            num_sampled_edges=[0, num_orders],
        ),
    }
    target = data.target(task).to_pandas()["target"]
    task_type = (
        TaskType.MULTICLASS_CLASSIFICATION
        if task == "classification"
        else TaskType.REGRESSION
    )
    anchor_time = (
        task_frame["task_time"]
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy()
    )
    return Context(
        task_type=task_type,
        entity_table_names=("customers",),
        subgraph=Subgraph(
            anchor_time=anchor_time,
            table_dict=table_dict,
            link_dict=link_dict,
        ),
        y_train=target,
        y_test=None,
        task_table=Table(
            df=task_frame,
            row=None,
            batch=batches,
            num_sampled_nodes=[],
            stype_dict={
                "task_score": KumoStype.numerical,
                "task_group": KumoStype.categorical,
                "task_time": KumoStype.timestamp,
            },
            primary_key=None,
        ),
    )


def _measure(
    operation: Callable[[], Any],
    *,
    task: Task,
    stage: Stage,
    device: torch.device,
    context_rows: int,
    query_rows: int,
    products: int,
    repetitions: int,
    warmups: int,
    profile_kernels: bool,
) -> ReferenceMeasurement:
    for _ in range(warmups):
        result = operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del result

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
    return ReferenceMeasurement(
        task=task,
        device=str(device),
        stage=stage,
        context_rows=context_rows,
        query_rows=query_rows,
        products=products,
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
        cuda_event_elapsed_median_ms=(
            statistics.median(cuda_elapsed) if cuda_elapsed else None
        ),
        cuda_kernel_time_ms=kernel_time,
        peak_memory_bytes=peak_delta,
        absolute_peak_memory_bytes=absolute_peak,
        throughput_task_rows_per_second=(
            (context_rows + query_rows) / (median / 1000)
        ),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the exact-reference benchmark matrix."""
    if git_version != REFERENCE_COMMIT:
        raise RuntimeError(
            f"Expected kumo-ml commit {REFERENCE_COMMIT}, got {git_version}."
        )
    measurements = []
    defaults = {}
    for task in args.tasks:
        task_type = (
            TaskType.MULTICLASS_CLASSIFICATION
            if task == "classification"
            else TaskType.REGRESSION
        )
        defaults[task] = repr(InferenceConfig.from_task_type(task_type))

    for device_name in args.devices:
        device = torch.device(device_name)
        driver = KumoRFMDriverV2_1(device=device)
        for task in args.tasks:
            context = build_context(
                task,
                context_rows=args.context_rows,
                query_rows=args.query_rows,
                products=args.products,
            )
            operations: dict[Stage, Callable[[], Any]] = {
                "data_from_context": functools.partial(
                    Data.from_context,
                    context,
                    infer_encoders=True,
                    device=device,
                ),
                "predict": functools.partial(
                    driver.predict,
                    context,
                    run_mode=RunMode.FAST,
                ),
            }
            for stage in args.stages:
                measurements.append(
                    _measure(
                        operations[stage],
                        task=task,
                        stage=stage,
                        device=device,
                        context_rows=args.context_rows,
                        query_rows=args.query_rows,
                        products=args.products,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        profile_kernels=args.profile_kernels,
                    )
                )

    return {
        "metadata": {
            **_metadata(),
            "kumo_ml_commit": git_version,
            "kumo_ml_version": version("kumo-ml"),
            "kumo_api_version": version("kumo-api"),
            "default_run_mode": str(RunMode.FAST),
            "default_configs": defaults,
        },
        "configuration": {
            "tasks": args.tasks,
            "devices": args.devices,
            "stages": args.stages,
            "context_rows": args.context_rows,
            "query_rows": args.query_rows,
            "products": args.products,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "profile_kernels": args.profile_kernels,
        },
        "results": [asdict(measurement) for measurement in measurements],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("classification", "regression"),
        default=["classification", "regression"],
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("data_from_context", "predict"),
        default=["data_from_context", "predict"],
    )
    parser.add_argument("--context-rows", type=int, default=1_000)
    parser.add_argument("--query-rows", type=int, default=1_000)
    parser.add_argument("--products", type=int, default=1_000)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/rfm_kumo_reference.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark and write JSON output."""
    args = parse_args(argv)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
