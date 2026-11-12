# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the actual Kumo recipe without constructing or running a model."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self, TypeVar, cast

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, EnsembleTable, StringTensor, TableTensor
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.processing import Processor
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)

T = TypeVar("T")

MATRIX = (
    (1000, 100),
    (1000, 500),
    (1000, 2000),
    (10000, 500),
    (10000, 2000),
    (100000, 100),
)
# SDMKumoTabularModel.default_num_estimators; direct inference defaults to 1.
BASE_MEMBERS = 16
SEED = 7971344


def _rss() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


class _Memory:
    def __enter__(self) -> Self:
        self.start = self.peak = _rss()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self) -> None:
        while not self.stop.wait(0.002):
            self.peak = max(self.peak, _rss())

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, _rss())
        self.stop.set()
        self.thread.join()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    samples = sorted(samples)
    p = (len(samples) - 1) * 0.95
    low = int(p)
    high = min(low + 1, len(samples) - 1)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": samples[low] + (p - low) * (samples[high] - samples[low]),
        "samples_ms": samples,
    }


def _fixture(path: Path, rows: int, columns: int) -> dict[str, Any]:
    if path.exists():
        return torch.load(path, weights_only=False)
    generator = torch.Generator().manual_seed(SEED + rows + columns)
    numerical_columns, categorical_columns, text_columns = (
        columns * 89 // 100,
        columns // 10,
        columns // 100,
    )
    parts = []
    for query in (False, True):
        numerical = torch.randn(rows, numerical_columns, generator=generator)
        numerical[:, 1::4].exp_()
        numerical[:, 2::4].mul_(4)
        numerical[::37, ::13] = torch.nan
        numerical[:, -1] = 2.0
        codes = torch.randint(
            0, 64, (rows, categorical_columns), generator=generator
        )
        codes[:, ::2].remainder_(12)
        codes[::43] = -1
        # Query dictionaries are deliberately reversed, not merely reused.
        categories = [
            tuple(
                f"c{column}_{index:03}"
                for index in range(12 if column % 2 == 0 else 64)
            )
            for column in range(categorical_columns)
        ]
        if query:
            categories = [tuple(reversed(values)) for values in categories]
        text = torch.randint(0, 8, (rows, text_columns), generator=generator)
        target = torch.randint(0, 5, (rows, 1), generator=generator)
        parts.append(
            {
                "numerical": numerical,
                "codes": codes,
                "categories": categories,
                "text": text,
                "target": target,
            }
        )
    fixture = {
        "rows": rows,
        "columns": columns,
        "stypes": {
            "numerical": numerical_columns,
            "categorical": categorical_columns,
            "text": text_columns,
        },
        "parts": parts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, path)
    return fixture


def _table(
    part: dict[str, Any], device: torch.device
) -> tuple[TableTensor, TableTensor]:
    texts = (
        "small blue table",
        "fresh green leaf",
        "wide river valley",
        "bright summer sun",
        "quiet winter morning",
        "deep forest path",
        "red stone bridge",
        "quick brown fox",
    )
    text = StringTensor.from_list(
        [[texts[index] for index in row] for row in part["text"].tolist()]
    )
    categorical = CategoricalTensor(
        code=part["codes"].to(torch.int32),
        categories=tuple(
            StringTensor.from_list(values) for values in part["categories"]
        ),
    )
    table = TableTensor(
        numerical=part["numerical"], categorical=categorical, text=text
    )
    target = TableTensor(
        categorical=CategoricalTensor(
            code=part["target"].to(torch.int32), categories=(torch.arange(5),)
        )
    )
    return cast(TableTensor, table.to(device)), cast(
        TableTensor, target.to(device)
    )


def _digest(fixture: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for part in fixture["parts"]:
        for key in ("numerical", "codes", "text", "target"):
            digest.update(part[key].numpy().tobytes())
        digest.update(json.dumps(part["categories"]).encode())
    return digest.hexdigest()


def _invoke(
    processor: Processor, operation: str, table: EnsembleTable, **kwargs: Any
) -> Any:
    # The harness deliberately survives removal of the old ensemble API.
    legacy = getattr(processor, operation + "_ensemble", None)
    return (legacy if legacy is not None else getattr(processor, operation))(
        table, **kwargs
    )


def _generator(device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(SEED)


def _pipeline(
    x: TableTensor,
    y: TableTensor,
    query: TableTensor,
    members: int,
    device: torch.device,
) -> tuple[
    RecipeExecution, tuple[MemberContext, ...], tuple[MemberQuery, ...]
]:
    execution = RecipeExecution(default_recipe())
    generator = _generator(device)
    contexts = execution.fit_transform(
        x=x, y=y, related_tables=None, num_members=members, generator=generator
    )
    queries = execution.transform(x=query, related_tables=None)
    return execution, contexts, queries


def _measure(
    action: Callable[[], T], device: torch.device
) -> tuple[T, dict[str, Any]]:
    _sync(device)
    gpu_start = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _Memory() as memory:
        started = time.perf_counter()
        result = action()
        _sync(device)
        elapsed = (time.perf_counter() - started) * 1000
    return result, {
        "ms": elapsed,
        "cpu_peak_rss_bytes": memory.peak,
        "cpu_rss_increase_bytes": max(memory.peak - memory.start, 0),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "gpu_allocation_increase_bytes": max(
            torch.cuda.max_memory_allocated(device) - gpu_start, 0
        )
        if device.type == "cuda"
        else 0,
    }


def _processor_classes() -> set[type[Processor]]:
    return {
        value
        for value in vars(sp).values()
        if inspect.isclass(value) and issubclass(value, Processor)
    }


class _Trace:
    """Separate instrumented run; none of its overhead is in total timings."""

    def __init__(
        self, device: torch.device, *, measure_memory: bool = False
    ) -> None:
        self.device = device
        self.measure_memory = measure_memory
        self.records = []
        self.stack = []
        self.patches = []

    def _sample_memory(self, *, gpu: bool = True) -> None:
        rss = _rss()
        peak = (
            torch.cuda.max_memory_allocated(self.device)
            if gpu and self.device.type == "cuda"
            else 0
        )
        for record in tuple(self.stack):
            record["cpu_peak_rss_bytes"] = max(
                record["cpu_peak_rss_bytes"], rss
            )
            record["gpu_peak_allocated_bytes"] = max(
                record["gpu_peak_allocated_bytes"], peak
            )

    def _sample(self) -> None:
        while not self.stop.wait(0.002):
            self._sample_memory(gpu=False)

    def __enter__(self) -> Self:
        tracer = self
        for cls in _processor_classes():
            for name in (
                "_fit",
                "_transform",
                "_fit_transform",
                "_fit_ensemble",
                "_transform_ensemble",
                "_fit_transform_ensemble",
            ):
                function = cls.__dict__.get(name)
                if function is None or not inspect.isfunction(function):
                    continue

                def wrapped(
                    self: Processor,
                    *args: Any,
                    _function: Callable[..., Any] = function,
                    _name: str = name,
                    **kwargs: Any,
                ) -> Any:
                    label = repr(self)
                    if len(label) > 180:
                        label = type(self).__name__
                    record: dict[str, Any] = {
                        "processor": label,
                        "operation": _name.replace("_ensemble", "").lstrip(
                            "_"
                        ),
                        "cpu_ms": 0.0,
                        "nested_cpu_ms": 0.0,
                        "calls": 1,
                        "ops": Counter(),
                        "allocated_bytes": Counter(),
                        "allocations": Counter(),
                    }
                    if args and isinstance(args[0], EnsembleTable):
                        record.update(
                            logical_members=len(args[0]),
                            physical_groups=args[0].num_groups,
                            physical_values=sum(
                                group.size(0)
                                for group in args[0]._iter_groups()
                            ),
                        )
                    elif args and isinstance(args[0], TableTensor):
                        record.update(
                            physical_executions=1,
                            physical_values=args[0].size(0)
                            if args[0].dim() > 2
                            else 1,
                        )
                    tracer.records.append(record)
                    record["_range_name"] = (
                        label
                        + "/"
                        + record["operation"]
                        + "#"
                        + str(len(tracer.records))
                    )
                    if tracer.measure_memory:
                        # Preserve parent peaks before resetting for a child.
                        tracer._sample_memory()
                        record["cpu_start_rss_bytes"] = _rss()
                        record["cpu_peak_rss_bytes"] = record[
                            "cpu_start_rss_bytes"
                        ]
                        record["gpu_start_allocated_bytes"] = (
                            torch.cuda.memory_allocated(tracer.device)
                            if tracer.device.type == "cuda"
                            else 0
                        )
                        record["gpu_peak_allocated_bytes"] = record[
                            "gpu_start_allocated_bytes"
                        ]
                        if tracer.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(tracer.device)
                    tracer.stack.append(record)
                    begin = end = None
                    if tracer.device.type == "cuda":
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        begin.record()
                    started = time.perf_counter()
                    try:
                        with torch.profiler.record_function(
                            record["_range_name"]
                        ):
                            return _function(self, *args, **kwargs)
                    finally:
                        record["cpu_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        if end is not None:
                            end.record()
                            record["_events"] = (begin, end)
                        if tracer.measure_memory:
                            tracer._sample_memory()
                        tracer.stack.pop()
                        if tracer.stack:
                            tracer.stack[-1]["nested_cpu_ms"] += record[
                                "cpu_ms"
                            ]

                self.patches.append((cls, name, function))
                setattr(cls, name, functools.wraps(function)(wrapped))
        if self.measure_memory:
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.measure_memory:
            self.stop.set()
            self.thread.join()
        for cls, name, function in reversed(self.patches):
            setattr(cls, name, function)
        _sync(self.device)
        for record in self.records:
            if self.measure_memory:
                record["cpu_rss_increase_bytes"] = max(
                    record["cpu_peak_rss_bytes"]
                    - record["cpu_start_rss_bytes"],
                    0,
                )
                record["gpu_allocation_increase_bytes"] = max(
                    record["gpu_peak_allocated_bytes"]
                    - record["gpu_start_allocated_bytes"],
                    0,
                )
            events = record.pop("_events", None)
            record["gpu_ms"] = (
                events[0].elapsed_time(events[1]) if events else 0.0
            )
            record["exclusive_cpu_ms"] = max(
                record["cpu_ms"] - record["nested_cpu_ms"], 0.0
            )


def _attribute_profile(
    profile: torch.profiler.profile, records: list[dict[str, Any]]
) -> None:
    ranges = {record["_range_name"]: record for record in records}
    for event in profile.events():
        if not event.name.startswith("aten::"):
            continue
        parent = event.cpu_parent
        while parent is not None and parent.name not in ranges:
            parent = parent.cpu_parent
        if parent is None:
            continue
        record = ranges[parent.name]
        record["ops"][event.name] += 1
        for device, size in (
            ("cpu", event.self_cpu_memory_usage),
            ("cuda", getattr(event, "self_device_memory_usage", 0)),
        ):
            if size > 0:
                record["allocations"][device] += 1
                record["allocated_bytes"][device] += size


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["processor"], record["operation"])].append(record)
    result = []
    for (processor, operation), items in sorted(groups.items()):
        result.append(
            {
                "processor": processor,
                "operation": operation,
                "calls": len(items),
                "cpu": _summary([item["cpu_ms"] for item in items]),
                "gpu": _summary([item["gpu_ms"] for item in items]),
                "exclusive_cpu_ms": sum(
                    item["exclusive_cpu_ms"] for item in items
                ),
                "physical_executions": sum(
                    item.get("physical_executions", 0) for item in items
                ),
                "physical_groups": sum(
                    item.get("physical_groups", 0) for item in items
                ),
                "logical_members": sum(
                    item.get("logical_members", 0) for item in items
                ),
                "physical_values": sum(
                    item.get("physical_values", 0) for item in items
                ),
                "ops": dict(sum((item["ops"] for item in items), Counter())),
                "dense_allocations": dict(
                    sum((item["allocations"] for item in items), Counter())
                ),
                "dense_allocated_bytes": dict(
                    sum((item["allocated_bytes"] for item in items), Counter())
                ),
            }
        )
    return result


def _profile(
    action: Callable[[], Any], device: torch.device, path: Path
) -> list[dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, profile_memory=True, record_shapes=True
    ) as profile:
        with _Trace(device):
            action()
        _sync(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(path))
    events = [
        {
            "name": event.key,
            "calls": event.count,
            "cpu_us": event.self_cpu_time_total,
            "gpu_us": getattr(event, "self_device_time_total", 0),
            "cpu_bytes": event.self_cpu_memory_usage,
            "gpu_bytes": getattr(event, "self_device_memory_usage", 0),
        }
        for event in profile.key_averages()
    ]
    if device.type == "cuda" and not any(event["gpu_us"] for event in events):
        raise RuntimeError("Profiler did not capture CUDA activity")
    return events


def _outputs(
    contexts: Sequence[MemberContext], queries: Sequence[MemberQuery]
) -> list[dict[str, TableTensor]]:
    return [
        {
            "x": cast(TableTensor, context.x.cpu()),
            "y": cast(TableTensor, context.y.cpu()),
            "query": cast(TableTensor, query.x.cpu()),
        }
        for context, query in zip(contexts, queries, strict=True)
    ]


def _parity(
    actual: Sequence[dict[str, TableTensor]],
    reference: Sequence[dict[str, TableTensor]],
) -> None:
    assert len(actual) == len(reference)
    for member, (left, right) in enumerate(
        zip(actual, reference, strict=True)
    ):
        for key in ("x", "y", "query"):
            assert left[key].column_groups == right[key].column_groups, (
                member,
                key,
                "columns",
            )
            torch.testing.assert_close(
                left[key].numerical,
                right[key].numerical,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            )
            assert left[key].categorical.equal(right[key].categorical), (
                member,
                key,
                "categories",
            )
            assert left[key].text.equal(right[key].text), (member, key, "text")


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    """Measure one recipe configuration against a fixed synthetic fixture."""
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    root = Path(args.artifacts).resolve()
    fixture = _fixture(
        root / "data" / f"{args.rows}x{args.columns}.pt",
        args.rows,
        args.columns,
    )
    x, y = _table(fixture["parts"][0], device)
    query, _ = _table(fixture["parts"][1], device)
    identifier = f"{args.rows}x{args.columns}-e{args.members}-{device.type}"
    if (
        args.phase == "baseline"
        and not args.parity_only
        and not args.counters_only
        and not args.profile_only
        and not args.memory_only
        and (root / "baseline" / f"{identifier}.json").exists()
    ):
        raise FileExistsError("Preserved baseline must not be overwritten")
    gpu_processes = ""
    if device.type == "cuda":
        gpu_processes = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device.index or 0}",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        foreign = [
            line
            for line in gpu_processes.splitlines()
            if int(line.split(",")[0]) != os.getpid()
        ]
        if foreign and not args.parity_only and not args.counters_only:
            raise RuntimeError(
                "GPU is occupied by other processes: " + "; ".join(foreign)
            )
    result = {
        "case": identifier,
        "phase": args.phase,
        "gpu_processes_pid_mib": gpu_processes,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset_sha256": _digest(fixture),
        "source_sha256": hashlib.sha256(
            b"".join(
                str(path).encode() + path.read_bytes()
                for path in sorted(Path("sdm").rglob("*.py"))
            )
        ).hexdigest(),
        "stypes": fixture["stypes"],
        "rows": args.rows,
        "query_rows": args.rows,
        "columns": args.columns,
        "members": args.members,
        "seed": SEED,
        "threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "recipe": repr(default_recipe()),
        "memory_scope": (
            "RSS sampled every 2ms during calls; GPU allocated peaks include "
            "live inputs; deltas are relative to start of each call"
        ),
        "instrumentation_scope": (
            "Per-processor callbacks timed in separate instrumented runs; "
            "inclusive times must not be summed. ATen operator dispatches "
            "and allocations come from a lightweight CPU-only profiler pass "
            "without shapes or stacks. Full CPU/CUDA profiling is selective. "
            "Positive self-memory events count allocation-producing ATen "
            "events, not malloc calls. Physical values describe inputs, not "
            "proven arithmetic redundancy. Old table-body execution counts "
            "and new ensemble processor-body counts are not comparable; "
            "compare ATen dispatches and selective kernel traces instead."
        ),
    }
    if args.memory_only:
        for _ in range(args.warmups):
            _pipeline(x, y, query, args.members, device)
        with _Trace(device, measure_memory=True) as tracer:
            _, contexts, queries = _pipeline(x, y, query, args.members, device)
        _parity(
            _outputs(contexts, queries),
            torch.load(
                root / "parity" / f"{identifier}.pt", weights_only=False
            ),
        )
        peaks: dict[tuple[str, str], dict[str, Any]] = {}
        for record in tracer.records:
            key = record["processor"], record["operation"]
            row = peaks.setdefault(
                key,
                {"processor": key[0], "operation": key[1], "calls": 0},
            )
            row["calls"] += 1
            for field in (
                "cpu_peak_rss_bytes",
                "cpu_rss_increase_bytes",
                "gpu_peak_allocated_bytes",
                "gpu_allocation_increase_bytes",
            ):
                row[field] = max(row.get(field, 0), record[field])
        result["processor_memory"] = list(peaks.values())
        result["status"] = "processor memory and parity; not latency"
        directory = root / "memory" / args.phase
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{identifier}.json").write_text(
            json.dumps(result, indent=2)
        )
        print(json.dumps({"case": identifier, "status": result["status"]}))
        return
    if args.profile_only:
        action = functools.partial(
            _pipeline, x, y, query, args.members, device
        )
        for _ in range(args.warmups):
            action()
        directory = root / "profiles" / args.phase
        result["profile"] = _profile(
            action, device, directory / f"{identifier}-trace.json"
        )
        result["status"] = "profile only; not a latency measurement"
        (directory / f"{identifier}.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            json.dumps({"case": identifier, "status": result["status"]}),
            flush=True,
        )
        return
    if args.parity_only or args.counters_only:
        if args.counters_only:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
            ) as profile:
                with _Trace(device) as tracer:
                    _, contexts, queries = _pipeline(
                        x, y, query, args.members, device
                    )
                _sync(device)
            _attribute_profile(profile, tracer.records)
            result["processor_counters"] = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"cpu", "gpu", "exclusive_cpu_ms"}
                }
                for record in _aggregate(tracer.records)
            ]
        else:
            _, contexts, queries = _pipeline(x, y, query, args.members, device)
        _parity(
            _outputs(contexts, queries),
            torch.load(
                root / "parity" / f"{identifier}.pt", weights_only=False
            ),
        )
        result["status"] = "parity passed; no timing measurements"
        path = (
            root
            / ("counters" if args.counters_only else "parity-checks")
            / f"{identifier}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(
            json.dumps({"case": identifier, "status": result["status"]}),
            flush=True,
        )
        return
    samples = defaultdict(list)
    memory = defaultdict(list)
    contexts = queries = None
    execution = None
    for repetition in range(args.warmups + args.repeats):
        execution = RecipeExecution(default_recipe())
        generator = _generator(device)
        (contexts, fit_memory) = _measure(
            functools.partial(
                execution.fit_transform,
                x=x,
                y=y,
                related_tables=None,
                num_members=args.members,
                generator=generator,
            ),
            device,
        )
        (queries, transform_memory) = _measure(
            functools.partial(
                execution.transform, x=query, related_tables=None
            ),
            device,
        )
        if repetition >= args.warmups:
            samples["fit_transform"].append(fit_memory.pop("ms"))
            samples["transform"].append(transform_memory.pop("ms"))
            samples["total"].append(
                samples["fit_transform"][-1] + samples["transform"][-1]
            )
            memory["fit_transform"].append(fit_memory)
            memory["transform"].append(transform_memory)
    result["effective_members"] = execution.num_members
    result["harness_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    result["platform"] = platform.platform()
    result["gpu_properties"] = (
        str(torch.cuda.get_device_properties(device))
        if device.type == "cuda"
        else None
    )
    result["dependency_heads"] = {
        "1344": "76909cd2e44f68dc13d303e4d6bf1642aed0e30c",
        "1361": "bfef597007f8c0aab43c539c8ccca697cadf0931",
    }
    result["latency"] = {
        name: _summary(values) for name, values in samples.items()
    }
    result["memory"] = dict(memory.items())
    assert contexts is not None
    assert queries is not None
    outputs = _outputs(contexts, queries)
    expected_path = root / "parity" / f"{identifier}.pt"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "baseline":
        if expected_path.exists():
            _parity(outputs, torch.load(expected_path, weights_only=False))
        else:
            torch.save(outputs, expected_path)
    else:
        _parity(outputs, torch.load(expected_path, weights_only=False))
    del outputs, contexts, queries, execution

    x_ensemble = EnsembleTable.from_table(x, num_members=args.members)
    fit_samples = []
    for repetition in range(args.warmups + args.repeats):
        recipe = default_recipe()
        _, measured = _measure(
            functools.partial(
                _invoke,
                recipe.features,
                "fit",
                x_ensemble,
                generator=_generator(device),
            ),
            device,
        )
        if repetition >= args.warmups:
            fit_samples.append(measured["ms"])
    result["latency"]["features_fit_unfused"] = _summary(fit_samples)
    del recipe

    records = []
    for _ in range(args.repeats):
        with _Trace(device) as tracer:
            _pipeline(x, y, query, args.members, device)
        records.extend(tracer.records)
    result["processors"] = _aggregate(records)

    if args.profile:
        # CUDA must be the first profiler session: initializing CPU-only
        # profiling first disables CUDA collection in this PyTorch build.
        result["profile"] = _profile(
            functools.partial(_pipeline, x, y, query, args.members, device),
            device,
            root / args.phase / f"{identifier}-trace.json",
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as light_profile:
        with _Trace(device) as light_trace:
            _pipeline(x, y, query, args.members, device)
        _sync(device)
    _attribute_profile(light_profile, light_trace.records)
    result["processor_counters"] = _aggregate(light_trace.records)

    path = root / args.phase / f"{identifier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "path": str(path),
                "latency": result["latency"],
                "effective_members": result["effective_members"],
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run one isolated case or the requested matrix in fresh processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("baseline", "after"), required=True
    )
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument("--members", type=int, default=BASE_MEMBERS)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--counters-only", action="store_true")
    parser.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    if not args.suite:
        run(args)
        return
    cases = [(rows, columns, BASE_MEMBERS) for rows, columns in MATRIX]
    cases += [(10000, 500, members) for members in (1, 4, 8)]
    if args.profile_only:
        cases = [(1000, 100, 16), (10000, 500, 16)]
    failures = []
    for rows, columns, members in cases:
        command = [
            os.sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            args.phase,
            "--artifacts",
            args.artifacts,
            "--device",
            args.device,
            "--rows",
            str(rows),
            "--columns",
            str(columns),
            "--members",
            str(members),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--threads",
            str(args.threads),
        ]
        if args.memory_only:
            command.append("--memory-only")
        elif args.profile_only:
            command.append("--profile-only")
        elif args.counters_only:
            command.append("--counters-only")
        elif args.parity_only:
            command.append("--parity-only")
        elif (rows, columns, members) in ((1000, 100, 16), (10000, 500, 16)):
            command.append("--profile")
        outcome = subprocess.run(command, check=False)
        if outcome.returncode:
            failures.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "members": members,
                    "returncode": outcome.returncode,
                }
            )
    if failures:
        path = Path(args.artifacts) / args.phase / "failures.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failures, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
