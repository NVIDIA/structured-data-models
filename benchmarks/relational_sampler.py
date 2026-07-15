"""Compare host PyG and CUDA cuGraph relational sampling on RelBench.

The runner intentionally keeps workload preparation separate from sampler
initialization and request timing. Run ``--mode cpu`` where ``pyg-lib`` is
available and ``--mode cuda`` in a RAPIDS environment, then use ``compare``
to combine the independently generated JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar, cast

import pyarrow as pa
import torch
from sdm import RelationalData, Stype, TableTensor, infer_stypes
from sdm.relational import CuGraphRelationalSampler, RelationalSampler
from sdm.relational.sampler import EXAMPLE_ID, RelationalSamplerOutput

RESULT_SCHEMA_VERSION = 2
DEFAULT_DATASET = "rel-arxiv"
DEFAULT_TASK = "paper-citation"
DEFAULT_BATCH_SIZES = (1, 128, 1024)
DEFAULT_FANOUTS = ((16,), (16, 16))
PHASE_ORDER = (
    "task_to_seed_join_ms",
    "neighbor_sampling_ms",
    "temporal_top_k_ms",
    "output_assembly_ms",
    "synchronization_other_ms",
)
T = TypeVar("T")


@dataclass(frozen=True)
class Workload:
    """RelBench data and task inputs shared by both sampler implementations."""

    data: RelationalData
    task_table: TableTensor
    task_link: dict[str, str]
    time_columns: dict[str, str]
    task_time_column: str
    identity_columns: dict[str, tuple[str, ...]]
    identity_multiplicities: dict[str, Counter[Any]]
    contract: dict[str, Any]


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of positive integers."""
    try:
        parsed = tuple(int(part) for part in value.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got '{value}'"
        ) from exc
    if not parsed or any(number <= 0 for number in parsed):
        raise argparse.ArgumentTypeError(
            "Expected at least one positive integer"
        )
    return parsed


def parse_fanout(value: str) -> tuple[int, ...]:
    """Parse one comma-separated fanout vector; ``-1`` means exhaustive."""
    try:
        fanout = tuple(int(part) for part in value.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated fanouts, got '{value}'"
        ) from exc
    if not fanout or any(count < -1 for count in fanout):
        raise argparse.ArgumentTypeError(
            "Fanouts must be -1 (exhaustive) or non-negative integers"
        )
    return fanout


def workload_kind(fanout: Sequence[int]) -> str:
    """Classify exhaustive output parity and finite stochastic workloads."""
    if all(count == -1 for count in fanout):
        return "exhaustive_parity"
    return "finite_stochastic"


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("Expected at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("Expected 'quantile' in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize request measurements in milliseconds or row counts."""
    if not values:
        raise ValueError("Expected at least one measurement")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _identity_columns(table: Any) -> tuple[str, ...]:
    columns = [
        column
        for column in (
            table.pkey_col,
            *table.fkey_col_to_pkey_table,
        )
        if column is not None
    ]
    return tuple(dict.fromkeys(columns))


def relational_inputs_from_database(
    database: Any,
) -> tuple[RelationalData, dict[str, str], dict[str, tuple[str, ...]]]:
    """Convert a RelBench-like database to SDM relational inputs.

    RelBench foreign keys are explicitly marked as IDs. Naming heuristics alone
    are insufficient for datasets whose relational columns do not contain an
    ``id`` token.
    """
    tables: dict[str, TableTensor] = {}
    relationships: list[dict[str, str]] = []
    time_columns: dict[str, str] = {}
    identity_columns: dict[str, tuple[str, ...]] = {}

    for name, table in database.table_dict.items():
        identity = _identity_columns(table)
        identity_columns[name] = identity
        stypes = infer_stypes(
            table.df,
            overrides=dict.fromkeys(identity, Stype.id),
        )
        tables[name] = TableTensor.from_pandas(df=table.df, stypes=stypes)
        if table.time_col is not None:
            time_columns[name] = table.time_col

        for left_column, right_table in table.fkey_col_to_pkey_table.items():
            right_column = database.table_dict[right_table].pkey_col
            if right_column is None:
                raise ValueError(
                    f"Expected referenced table '{right_table}' to have a "
                    "primary key"
                )
            relationships.append(
                {
                    "left_table": name,
                    "left_column": left_column,
                    "right_table": right_table,
                    "right_column": right_column,
                }
            )

    return (
        RelationalData(tables=tables, relationships=relationships),
        (time_columns),
        identity_columns,
    )


def _task_stypes(task: Any) -> dict[str, Stype]:
    task_type = getattr(task.task_type, "name", str(task.task_type))
    target_stype = (
        Stype.numerical if task_type == "REGRESSION" else Stype.categorical
    )
    return {
        task.entity_col: Stype.id,
        task.time_col: Stype.datetime,
        task.target_col: target_stype,
    }


def _table_contract(
    database: Any,
    identity_columns: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    contract = []
    for name, table in database.table_dict.items():
        sampling_columns = list(identity_columns[name])
        if (
            table.time_col is not None
            and table.time_col not in sampling_columns
        ):
            sampling_columns.append(table.time_col)
        sampling_table = pa.Table.from_pandas(
            table.df[sampling_columns],
            preserve_index=False,
        )
        contract.append(
            {
                "name": name,
                "rows": len(table.df),
                "columns": list(table.df.columns),
                "identity_columns": list(identity_columns[name]),
                "time_column": table.time_col,
                "sampling_data_sha256": _hash_arrow(sampling_table),
            }
        )
    return contract


def build_relbench_workload(
    dataset_name: str,
    task_name: str,
    split: str,
) -> Workload:
    """Download/cache a RelBench dataset and construct the sampler workload."""
    try:
        import relbench
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
    except ImportError as exc:
        raise ImportError(
            "Install relbench to run this benchmark: pip install relbench"
        ) from exc

    database = get_dataset(dataset_name, download=True).get_db(
        upto_test_timestamp=False
    )
    data, time_columns, identity_columns = relational_inputs_from_database(
        database
    )
    task = get_task(dataset_name, task_name, download=True)
    required = ("entity_col", "entity_table", "time_col", "target_col")
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise ValueError(
            "This benchmark currently supports RelBench entity tasks; "
            f"missing {missing} for '{dataset_name}/{task_name}'"
        )
    task_table = TableTensor.from_pandas(
        df=task.get_table(split, mask_input_cols=False).df,
        stypes=_task_stypes(task),
    )
    task_link = {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(
            str, database.table_dict[task.entity_table].pkey_col
        ),
    }
    contract = {
        "dataset": dataset_name,
        "task": task_name,
        "split": split,
        "relbench_version": getattr(relbench, "__version__", "unknown"),
        "tables": _table_contract(database, identity_columns),
        "relationships": [
            {
                "left_table": relation.left_table,
                "left_columns": list(relation.left_columns),
                "right_table": relation.right_table,
                "right_columns": list(relation.right_columns),
            }
            for relation in data.relationships
        ],
        "time_columns": time_columns,
        "task_link": task_link,
        "task_time_column": task.time_col,
        "task_rows": task_table.size(0),
        "task_sampling_data_sha256": _hash_arrow(
            task_table.to_arrow().select((task.entity_col, task.time_col))
        ),
    }
    return Workload(
        data=data,
        task_table=task_table,
        task_link=task_link,
        time_columns=time_columns,
        task_time_column=task.time_col,
        identity_columns=identity_columns,
        identity_multiplicities={
            name: _rows_from_arrow(
                table.to_arrow(),
                identity_columns[name],
            )
            for name, table in data.tables.items()
        },
        contract=contract,
    )


def select_task_rows(
    task_table: TableTensor,
    batch_size: int,
    seed: int,
) -> tuple[TableTensor, list[int]]:
    """Choose task positions independently of the PyTorch RNG version."""
    if batch_size > task_table.size(0):
        raise ValueError(
            f"Batch size {batch_size} exceeds task rows {task_table.size(0)}"
        )
    positions = random.Random(seed).sample(
        range(task_table.size(0)), batch_size
    )
    index = torch.tensor(positions, dtype=torch.long)
    return cast(TableTensor, task_table[index]), positions


def task_selection_summary(positions: Sequence[int]) -> dict[str, Any]:
    """Return a compact, exact identity for a deterministic task selection."""
    encoded = json.dumps(list(positions), separators=(",", ":")).encode()
    return {
        "count": len(positions),
        "positions_sha256": hashlib.sha256(encoded).hexdigest(),
        "position_prefix": list(positions[:8]),
    }


def _hash_arrow(table: pa.Table) -> str:
    table = table.replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _canonical_table_hash(table: TableTensor, columns: Sequence[str]) -> str:
    arrow = table.to_arrow().select(columns)
    keys = [(column, "ascending") for column in columns]
    return _hash_arrow(arrow.sort_by(keys))


def _rows_from_arrow(table: pa.Table, columns: Sequence[str]) -> Counter[Any]:
    values = table.select(columns).to_pydict()
    return Counter(zip(*(values[column] for column in columns)))


def output_invariants(
    output: RelationalSamplerOutput,
    workload: Workload,
) -> dict[str, Any]:
    """Check output alignment, temporal cutoffs, and canonical row identity."""
    errors: list[str] = []
    related = output.related_tables.tables
    seed_table_name = workload.task_link["table"]
    task_columns = (EXAMPLE_ID, workload.task_link["task_column"])
    seed_columns = (EXAMPLE_ID, workload.task_link["table_column"])
    task_arrow = output.task_table.to_arrow()
    task_rows = output.task_table.size(0)

    if seed_table_name not in related:
        errors.append(f"missing seed table '{seed_table_name}'")
    else:
        seed_arrow = related[seed_table_name].to_arrow()
        missing_seeds = _rows_from_arrow(
            task_arrow, task_columns
        ) - _rows_from_arrow(seed_arrow, seed_columns)
        if missing_seeds:
            errors.append("seed table does not retain every task seed")

    cutoff_by_example: dict[Any, Any] = {}
    if workload.time_columns:
        task_values = task_arrow.select(
            (EXAMPLE_ID, workload.task_time_column)
        ).to_pydict()
        cutoff_by_example = dict(
            zip(
                task_values[EXAMPLE_ID],
                task_values[workload.task_time_column],
            )
        )

    temporal_rows_checked = 0
    temporal_violations = 0
    repeated_identity_rows = 0
    repeated_identities_by_table: dict[str, int] = {}
    source_multiplicity_inflation_rows = 0
    source_multiplicity_inflation_by_table: dict[str, int] = {}
    table_hashes: dict[str, str] = {}
    rows_by_table: dict[str, int] = {}
    digest = hashlib.sha256()
    for name, table in sorted(related.items()):
        rows_by_table[name] = table.size(0)
        identity = workload.identity_columns[name]
        columns = (EXAMPLE_ID, *identity)
        identity_counts = _rows_from_arrow(table.to_arrow(), columns)
        repeated = sum(count - 1 for count in identity_counts.values())
        repeated_identities_by_table[name] = repeated
        repeated_identity_rows += repeated
        source_counts = workload.identity_multiplicities[name]
        inflation = sum(
            max(0, count - source_counts[identity[1:]])
            for identity, count in identity_counts.items()
        )
        source_multiplicity_inflation_by_table[name] = inflation
        source_multiplicity_inflation_rows += inflation
        table_hash = _canonical_table_hash(table, columns)
        table_hashes[name] = table_hash
        digest.update(name.encode())
        digest.update(table_hash.encode())

        time_column = workload.time_columns.get(name)
        if time_column is None:
            continue
        values = table.to_arrow().select((EXAMPLE_ID, time_column)).to_pydict()
        for example, row_time in zip(
            values[EXAMPLE_ID], values[time_column], strict=True
        ):
            temporal_rows_checked += 1
            if row_time is not None and row_time > cutoff_by_example[example]:
                temporal_violations += 1

    if temporal_violations:
        errors.append(f"{temporal_violations} temporal cutoff violations")
    if source_multiplicity_inflation_rows:
        errors.append(
            f"{source_multiplicity_inflation_rows} rows exceed source "
            "identity multiplicity"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "task_rows": task_rows,
        "sampled_rows": sum(rows_by_table.values()),
        "rows_by_table": rows_by_table,
        "table_hashes": table_hashes,
        "canonical_output_sha256": digest.hexdigest(),
        "temporal_rows_checked": temporal_rows_checked,
        "temporal_violations": temporal_violations,
        "repeated_identity_rows": repeated_identity_rows,
        "repeated_identities_by_table": repeated_identities_by_table,
        "source_multiplicity_inflation_rows": (
            source_multiplicity_inflation_rows
        ),
        "source_multiplicity_inflation_by_table": (
            source_multiplicity_inflation_by_table
        ),
    }


class CudaPhaseTimer:
    """Measure sequential CUDA stream spans with CUDA events."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.spans: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}

    def measure(self, name: str, fn: Callable[[], T]) -> T:
        """Run ``fn`` between CUDA events on the sampler's current stream."""
        with torch.cuda.device(self.device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            value = fn()
            end.record()
        self.spans.setdefault(name, []).append((start, end))
        return value

    def finish(self) -> dict[str, float]:
        """Synchronize once and return accumulated event durations in ms."""
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
        return {
            name: sum(start.elapsed_time(end) for start, end in events)
            for name, events in self.spans.items()
        }


def sample_cuda_with_phases(
    sampler: CuGraphRelationalSampler,
    task_table: TableTensor,
    task_link: Mapping[str, str],
    fanout: Sequence[int],
    task_time_column: str,
) -> tuple[RelationalSamplerOutput, dict[str, float], float, dict[str, int]]:
    """Sample once and separate CUDA join, sampling, top-k, and assembly.

    The temporal top-k is timed by wrapping the sampler's existing selection
    helper. It remains part of the normal sampler call; the non-top-k neighbor
    duration is the enclosing CUDA event span less that nested event span.
    """
    device = sampler.data.device
    timer = CudaPhaseTimer(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    validated_link = sampler._validate_sample_inputs(
        task_table=task_table,
        task_link=task_link,
        task_time_column=task_time_column,
    )
    seed = timer.measure(
        "task_to_seed_join_ms",
        lambda: sampler._resolve_seed(task_table, validated_link),
    )
    seed_time = task_table[task_time_column].datetime.squeeze(-1)

    original_last_per_source = sampler._last_per_source
    selected_groups: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ] = []

    def timed_last_per_source(**kwargs: Any) -> torch.Tensor:
        selected = timer.measure(
            "temporal_top_k_ms",
            lambda: original_last_per_source(**kwargs),
        )
        selected_groups.append(
            (
                kwargs["batch"],
                kwargs["major"],
                kwargs["edge_type"],
                selected,
                kwargs["count"],
            )
        )
        return selected

    sampler._last_per_source = timed_last_per_source
    try:
        if sampler._num_edges == 0:
            nodes = timer.measure(
                "neighbor_sampling_ms",
                lambda: sampler._seed_nodes(seed, validated_link),
            )
        elif sampler.time_columns:
            neighbor_total = timer.measure(
                "neighbor_sampling_total_ms",
                lambda: sampler._sample_temporal(
                    seed=seed,
                    seed_time=seed_time,
                    num_neighbors=fanout,
                ),
            )
            nodes = neighbor_total
        else:
            nodes = timer.measure(
                "neighbor_sampling_ms",
                lambda: sampler._sample_non_temporal(seed, fanout),
            )
    finally:
        del sampler._last_per_source

    output = timer.measure(
        "output_assembly_ms",
        lambda: sampler._to_output(task_table, validated_link, nodes),
    )
    phases = timer.finish()
    total_ms = (time.perf_counter_ns() - started) / 1_000_000
    if "neighbor_sampling_total_ms" in phases:
        total = phases.pop("neighbor_sampling_total_ms")
        top_k = phases.get("temporal_top_k_ms", 0.0)
        phases["neighbor_sampling_ms"] = max(0.0, total - top_k)
    device_ms = sum(phases.values())
    phases["synchronization_other_ms"] = max(0.0, total_ms - device_ms)
    groups_checked = 0
    max_selected_per_group = 0
    fanout_violations = 0
    for batch, major, edge_type, selected, count in selected_groups:
        if selected.numel() == 0:
            continue
        groups = torch.stack(
            (batch[selected], major[selected], edge_type[selected]),
            dim=1,
        )
        counts = groups.unique(dim=0, return_counts=True)[1]
        groups_checked += counts.numel()
        max_selected_per_group = max(
            max_selected_per_group,
            int(counts.max().item()),
        )
        fanout_violations += int((counts > count).sum().item())
    return (
        output,
        phases,
        total_ms,
        {
            "groups_checked": groups_checked,
            "max_selected_per_group": max_selected_per_group,
            "violations": fanout_violations,
        },
    )


def _sample_cuda_once(
    sampler: CuGraphRelationalSampler,
    task_table: TableTensor,
    task_link: Mapping[str, str],
    fanout: Sequence[int],
    task_time_column: str,
) -> tuple[RelationalSamplerOutput, float]:
    torch.cuda.synchronize(task_table.device)
    started = time.perf_counter_ns()
    output = sampler(
        task_table=task_table,
        task_link=task_link,
        num_neighbors=fanout,
        task_time_column=task_time_column,
    )
    torch.cuda.synchronize(task_table.device)
    return output, (time.perf_counter_ns() - started) / 1_000_000


def _sample_cpu_once(
    sampler: RelationalSampler,
    task_table: TableTensor,
    task_link: Mapping[str, str],
    fanout: Sequence[int],
    task_time_column: str,
) -> tuple[RelationalSamplerOutput, float]:
    started = time.perf_counter_ns()
    output = sampler(
        task_table=task_table,
        task_link=task_link,
        num_neighbors=fanout,
        task_time_column=task_time_column,
    )
    return output, (time.perf_counter_ns() - started) / 1_000_000


def _initialize_sampler(
    mode: str,
    workload: Workload,
    random_state: int,
) -> tuple[RelationalSampler | CuGraphRelationalSampler, float, float | None]:
    if mode == "cpu":
        started = time.perf_counter_ns()
        sampler = RelationalSampler(
            data=workload.data,
            time_columns=workload.time_columns,
        )
        return sampler, (time.perf_counter_ns() - started) / 1_000_000, None

    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    transfer_started = time.perf_counter_ns()
    cuda_data = workload.data.to(device)
    torch.cuda.synchronize(device)
    transfer_ms = (time.perf_counter_ns() - transfer_started) / 1_000_000
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    sampler = CuGraphRelationalSampler(
        data=cuda_data,
        time_columns=workload.time_columns,
        random_state=random_state,
    )
    torch.cuda.synchronize(device)
    return sampler, (time.perf_counter_ns() - started) / 1_000_000, transfer_ms


def _cpu_environment() -> dict[str, Any]:
    fields: dict[str, str] = {}
    try:
        output = subprocess.check_output(
            ["lscpu", "--json"],
            text=True,
        )
        entries = json.loads(output)["lscpu"]
        fields = {
            entry["field"].rstrip(":"): entry["data"] for entry in entries
        }
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        KeyError,
    ):
        pass

    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    load_average = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    return {
        "model": fields.get("Model name", platform.processor() or "unknown"),
        "architecture": fields.get("Architecture", platform.machine()),
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "load_average_1m_5m_15m": load_average,
        "sockets": fields.get("Socket(s)"),
        "cores_per_socket": fields.get("Core(s) per socket"),
        "threads_per_core": fields.get("Thread(s) per core"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _environment() -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    packages = {
        "cudf": "cudf",
        "cupy": "cupy",
        "numpy": "numpy",
        "pyarrow": "pyarrow",
        "pyg_lib": "pyg-lib",
        "pylibcugraph": "pylibcugraph",
        "relbench": "relbench",
    }
    for module_name, package_name in packages.items():
        try:
            module = __import__(module_name)
        except ImportError:
            versions[module_name] = "not_installed"
        else:
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = importlib.metadata.version(package_name)
                except importlib.metadata.PackageNotFoundError:
                    version = "unknown"
            versions[module_name] = version
    distribution_versions = {}
    for package_name in (
        "cudf-cu13",
        "cupy-cuda13x",
        "pyg-lib",
        "pylibcugraph-cu13",
        "relbench",
        "torch",
    ):
        try:
            distribution_versions[package_name] = importlib.metadata.version(
                package_name
            )
        except importlib.metadata.PackageNotFoundError:
            distribution_versions[package_name] = "not_installed"
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = "unavailable"
    return {
        "versions": versions,
        "distribution_versions": distribution_versions,
        "cpu": _cpu_environment(),
        "gpu": gpu,
        "nvidia_driver": driver,
    }


def _run_variant(
    mode: str,
    sampler: RelationalSampler | CuGraphRelationalSampler,
    workload: Workload,
    batch_size: int,
    fanout: tuple[int, ...],
    warmups: int,
    repetitions: int,
    selection_seed: int,
) -> dict[str, Any]:
    task_table, positions = select_task_rows(
        workload.task_table,
        batch_size=batch_size,
        seed=selection_seed,
    )
    task_to_device_ms: float | None = None
    if mode == "cuda":
        torch.cuda.synchronize()
        transfer_started = time.perf_counter_ns()
        task_table = cast(TableTensor, task_table.to("cuda"))
        torch.cuda.synchronize(task_table.device)
        task_to_device_ms = (
            time.perf_counter_ns() - transfer_started
        ) / 1_000_000

    for _ in range(warmups):
        if mode == "cpu":
            assert isinstance(sampler, RelationalSampler)
            _sample_cpu_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )
        else:
            assert isinstance(sampler, CuGraphRelationalSampler)
            _sample_cuda_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )

    latencies: list[float] = []
    for _ in range(repetitions):
        if mode == "cpu":
            assert isinstance(sampler, RelationalSampler)
            output, latency = _sample_cpu_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )
        else:
            assert isinstance(sampler, CuGraphRelationalSampler)
            output, latency = _sample_cuda_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )
        del output
        latencies.append(latency)

    sampled_rows: list[float] = []
    output_to_host_ms: list[float] = []
    validation_fingerprint_ms: list[float] = []
    output_digests: list[str] = []
    invariant: dict[str, Any] | None = None
    for _ in range(repetitions):
        if mode == "cpu":
            assert isinstance(sampler, RelationalSampler)
            output, _ = _sample_cpu_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )
        else:
            assert isinstance(sampler, CuGraphRelationalSampler)
            output, _ = _sample_cuda_once(
                sampler,
                task_table,
                workload.task_link,
                fanout,
                workload.task_time_column,
            )
        materialization_started = time.perf_counter_ns()
        materialized_output = output.cpu()
        output_to_host_ms.append(
            (time.perf_counter_ns() - materialization_started) / 1_000_000
        )
        validation_started = time.perf_counter_ns()
        invariant = output_invariants(materialized_output, workload)
        validation_fingerprint_ms.append(
            (time.perf_counter_ns() - validation_started) / 1_000_000
        )
        if not invariant["valid"]:
            raise RuntimeError(
                f"Sampled-output invariant failed: {invariant['errors']}"
            )
        sampled_rows.append(float(invariant["sampled_rows"]))
        output_digests.append(invariant["canonical_output_sha256"])

    profiled_latencies: list[float] = []
    phase_samples: dict[str, list[float]] = {}
    fanout_observations: list[dict[str, int]] = []
    profiled_invariant: dict[str, Any] | None = None
    if mode == "cuda":
        assert isinstance(sampler, CuGraphRelationalSampler)
        for _ in range(repetitions):
            profiled_output, phases, profiled_latency, fanout_observation = (
                sample_cuda_with_phases(
                    sampler,
                    task_table,
                    workload.task_link,
                    fanout,
                    workload.task_time_column,
                )
            )
            profiled_latencies.append(profiled_latency)
            fanout_observations.append(fanout_observation)
            for name, value in phases.items():
                phase_samples.setdefault(name, []).append(value)

        profiled_invariant = output_invariants(profiled_output, workload)
        if not profiled_invariant["valid"]:
            raise RuntimeError(
                "Profiled sampled-output invariant failed: "
                f"{profiled_invariant['errors']}"
            )
        fanout_violations = sum(
            observation["violations"] for observation in fanout_observations
        )
        if fanout_violations:
            raise RuntimeError(
                f"Observed {fanout_violations} finite fanout violations"
            )

    latency = summarize(latencies)
    result = {
        "kind": workload_kind(fanout),
        "fanout": list(fanout),
        "batch_size": batch_size,
        "warmups": warmups,
        "repetitions": repetitions,
        "task_selection": task_selection_summary(positions),
        "latency_ms": latency,
        "throughput": {
            "requests_per_second": 1_000 / float(latency["median"]),
            "task_rows_per_second": batch_size
            * 1_000
            / float(latency["median"]),
        },
        "task_to_device_ms_excluded": task_to_device_ms,
        "output_to_host_ms_excluded": summarize(output_to_host_ms),
        "validation_fingerprint_ms_excluded": summarize(
            validation_fingerprint_ms
        ),
        "sampled_rows": summarize(sampled_rows),
        "invariants": invariant,
        "phases_ms": {
            name: summarize(values)
            for name, values in sorted(phase_samples.items())
        },
        "raw_samples": {
            "latency_ms": latencies,
            "sampled_rows": sampled_rows,
            "output_sha256": output_digests,
            "output_to_host_ms": output_to_host_ms,
            "validation_fingerprint_ms": validation_fingerprint_ms,
        },
    }
    if profiled_latencies:
        result["profiled_latency_ms"] = summarize(profiled_latencies)
        result["profiled_invariants"] = profiled_invariant
        result["raw_samples"].update(
            {
                "profiled_latency_ms": profiled_latencies,
                "phases_ms": phase_samples,
                "finite_fanout_observations": fanout_observations,
            }
        )
        groups_checked = sum(
            observation["groups_checked"]
            for observation in fanout_observations
        )
        if groups_checked:
            result["finite_fanout_observations"] = {
                "scope": (
                    "CUDA temporal latest-k groups observed during the "
                    "separate profiled pass"
                ),
                "groups_checked": groups_checked,
                "max_selected_per_group": max(
                    observation["max_selected_per_group"]
                    for observation in fanout_observations
                ),
                "violations": 0,
            }
    return result


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested request shapes in exactly one sampler mode."""
    torch.manual_seed(args.seed)
    workload = build_relbench_workload(
        dataset_name=args.dataset,
        task_name=args.task,
        split=args.split,
    )
    sampler, initialization_ms, transfer_ms = _initialize_sampler(
        mode=args.mode,
        workload=workload,
        random_state=args.seed,
    )
    fanouts = tuple(args.fanout) if args.fanout else DEFAULT_FANOUTS
    runs = [
        _run_variant(
            mode=args.mode,
            sampler=sampler,
            workload=workload,
            batch_size=batch_size,
            fanout=fanout,
            warmups=args.warmups,
            repetitions=args.repetitions,
            selection_seed=args.seed,
        )
        for fanout in fanouts
        for batch_size in args.batch_size
    ]
    if args.include_exhaustive:
        runs.append(
            _run_variant(
                mode=args.mode,
                sampler=sampler,
                workload=workload,
                batch_size=args.parity_batch_size,
                fanout=(-1,),
                warmups=args.warmups,
                repetitions=args.repetitions,
                selection_seed=args.seed,
            )
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "command": sys.argv,
        "environment": _environment(),
        "workload": workload.contract,
        "random_state": args.seed,
        "measurement": {
            "latency_ms": (
                "Synchronized public sampler call with task input already "
                "resident on the sampler device; validation runs in a "
                "separate untimed pass"
            ),
            "task_to_device_ms_excluded": (
                "One observed task-table transfer, excluded from latency_ms"
            ),
            "profiled_latency_ms": (
                "Separate CUDA pass with event instrumentation; "
                "phases are independent medians"
            ),
            "output_to_host_ms_excluded": (
                "Separate validation pass; materializes the returned output "
                "on the host after the sampler call"
            ),
            "validation_fingerprint_ms_excluded": (
                "Separate validation pass after host materialization; checks "
                "correctness and computes canonical output fingerprints"
            ),
        },
        "initialization": {
            "sampler_topology_build_ms": initialization_ms,
            "data_to_cuda_ms_excluded_from_init": transfer_ms,
        },
        "runs": runs,
    }


def _run_key(run: Mapping[str, Any]) -> tuple[str, tuple[int, ...], int]:
    return run["kind"], tuple(run["fanout"]), int(run["batch_size"])


def compare_results(
    cpu_result: Mapping[str, Any],
    cuda_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge independent mode results and assert exhaustive output parity."""
    if cpu_result["mode"] != "cpu" or cuda_result["mode"] != "cuda":
        raise ValueError("Expected a CPU result and a CUDA result")
    if cpu_result["workload"] != cuda_result["workload"]:
        raise ValueError("CPU and CUDA workload contracts differ")
    if cpu_result["random_state"] != cuda_result["random_state"]:
        raise ValueError("CPU and CUDA random-state seeds differ")
    cpu_versions = cpu_result["environment"]["versions"]
    cuda_versions = cuda_result["environment"]["versions"]
    if cpu_versions != cuda_versions:
        raise ValueError("CPU and CUDA runtime versions differ")
    cpu_distributions = cpu_result["environment"].get(
        "distribution_versions", {}
    )
    cuda_distributions = cuda_result["environment"].get(
        "distribution_versions", {}
    )
    if cpu_distributions != cuda_distributions:
        raise ValueError("CPU and CUDA distribution versions differ")
    cpu_runs = {_run_key(run): run for run in cpu_result["runs"]}
    cuda_runs = {_run_key(run): run for run in cuda_result["runs"]}
    common = sorted(cpu_runs.keys() & cuda_runs.keys())
    if not common:
        raise ValueError(
            "CPU and CUDA results have no common request variants"
        )

    comparisons: list[dict[str, Any]] = []
    parity_failures: list[str] = []
    parity_checked = False
    for key in common:
        cpu_run = cpu_runs[key]
        cuda_run = cuda_runs[key]
        if cpu_run["task_selection"] != cuda_run["task_selection"]:
            raise ValueError(f"CPU and CUDA task selections differ for {key}")
        if (
            cpu_run["warmups"] != cuda_run["warmups"]
            or cpu_run["repetitions"] != cuda_run["repetitions"]
        ):
            raise ValueError(
                f"CPU and CUDA measurement counts differ for {key}"
            )
        cpu_latency = float(cpu_run["latency_ms"]["median"])
        cuda_latency = float(cuda_run["latency_ms"]["median"])
        cpu_rows = float(cpu_run["sampled_rows"]["median"])
        cuda_rows = float(cuda_run["sampled_rows"]["median"])
        entry = {
            "kind": key[0],
            "fanout": list(key[1]),
            "batch_size": key[2],
            "cpu_median_ms": cpu_latency,
            "cuda_median_ms": cuda_latency,
            "cuda_speedup": cpu_latency / cuda_latency,
            "cpu_sampled_rows_median": cpu_rows,
            "cuda_sampled_rows_median": cuda_rows,
            "equal_sampled_row_count": cpu_rows == cuda_rows,
            "cpu_sampled_rows_per_second": cpu_rows * 1_000 / cpu_latency,
            "cuda_sampled_rows_per_second": cuda_rows * 1_000 / cuda_latency,
            "cuda_sampled_row_throughput_ratio": (
                (cuda_rows / cuda_latency) / (cpu_rows / cpu_latency)
            ),
            "cuda_task_to_device_ms_excluded": cuda_run.get(
                "task_to_device_ms_excluded"
            ),
            "cpu_output_to_host_median_ms_excluded": cpu_run[
                "output_to_host_ms_excluded"
            ]["median"],
            "cuda_output_to_host_median_ms_excluded": cuda_run[
                "output_to_host_ms_excluded"
            ]["median"],
            "cpu_validation_fingerprint_median_ms_excluded": cpu_run[
                "validation_fingerprint_ms_excluded"
            ]["median"],
            "cuda_validation_fingerprint_median_ms_excluded": cuda_run[
                "validation_fingerprint_ms_excluded"
            ]["median"],
        }
        if "profiled_latency_ms" in cuda_run:
            profiled = float(cuda_run["profiled_latency_ms"]["median"])
            entry["cuda_profiled_median_ms"] = profiled
            entry["cuda_profiled_minus_public_ms"] = profiled - cuda_latency
        if key[0] == "exhaustive_parity":
            parity_checked = True
            equal = (
                cpu_run["invariants"]["canonical_output_sha256"]
                == cuda_run["invariants"]["canonical_output_sha256"]
            )
            entry["exact_output_parity"] = equal
            if not equal:
                parity_failures.append(str(key))
        comparisons.append(entry)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": cpu_result["workload"],
        "random_state": cpu_result["random_state"],
        "runtime_versions_match": True,
        "cpu": {
            "environment": cpu_result["environment"],
            "initialization": cpu_result["initialization"],
        },
        "cuda": {
            "environment": cuda_result["environment"],
            "initialization": cuda_result["initialization"],
        },
        "comparisons": comparisons,
        "exhaustive_parity_checked": parity_checked,
        "exhaustive_parity_passed": (
            not parity_failures if parity_checked else None
        ),
        "exhaustive_parity_failures": parity_failures,
    }


def _timeline_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for run in result["runs"]:
        medians = {
            name: values["median"] for name, values in run["phases_ms"].items()
        }
        phases = {
            name: medians[name] for name in PHASE_ORDER if name in medians
        }
        phases.update(
            {
                name: medians[name]
                for name in sorted(medians.keys() - phases.keys())
            }
        )
        if phases:
            runs.append(
                {
                    "label": (
                        f"fanout={run['fanout']}, batch={run['batch_size']}"
                    ),
                    "public_latency_ms": run["latency_ms"]["median"],
                    "profiled_latency_ms": run.get(
                        "profiled_latency_ms",
                        run["latency_ms"],
                    )["median"],
                    "phases_ms": phases,
                }
            )
    return {"mode": result["mode"], "runs": runs}


def write_timeline_artifacts(
    result: Mapping[str, Any],
    html_path: Path,
    trace_path: Path,
) -> None:
    """Write a trace-compatible aggregate and an HTML phase profile."""
    payload = _timeline_payload(result)
    trace_events: list[dict[str, Any]] = []
    for process, run in enumerate(payload["runs"]):
        offset_us = 0.0
        for name, duration_ms in run["phases_ms"].items():
            trace_events.append(
                {
                    "name": name,
                    "ph": "X",
                    "pid": process,
                    "tid": 0,
                    "ts": offset_us,
                    "dur": duration_ms * 1_000,
                    "args": {
                        "request": run["label"],
                        "aggregation": "independent phase median",
                    },
                }
            )
            offset_us += duration_ms * 1_000
    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": (
                "Synthetic aggregate of independent phase medians in "
                "data-flow order; not one request execution timeline"
            ),
            "payload": payload,
        },
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")

    escaped_payload = json.dumps(payload).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>Relational sampler aggregate phase profile</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #18202a; }}
.run {{ margin: 20px 0 30px; }} .bar {{ display: flex; height: 34px; }}
.phase {{ min-width: 1px; box-sizing: border-box; border-right: 1px solid #fff;
  padding: 9px 4px; overflow: hidden; white-space: nowrap; font-size: 12px; }}
.legend {{ display: flex; gap: 12px; flex-wrap: wrap; font-size: 13px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px;
  margin-right: 4px; }}
</style><body><h1>CUDA relational sampler aggregate phase profile</h1>
<p>Each segment is an independently computed median, arranged in data-flow
order. This is a flamegraph-like latency breakdown, not a single-request
execution timeline. Synchronization/other is the synchronized wall-clock
residual from the profiled call.</p><div id="root"></div>
<script>const payload = {escaped_payload};
const colors = ['#1677b8','#24a37b','#df8b27','#9a5fb4','#607d8b'];
const root = document.getElementById('root');
payload.runs.forEach((run) => {{
  const total = Object.values(run.phases_ms).reduce((a, b) => a + b, 0);
  const item = document.createElement('section'); item.className = 'run';
  const publicLatency = run.public_latency_ms.toFixed(3);
  const profiledLatency = run.profiled_latency_ms.toFixed(3);
  item.innerHTML = `<h2>${{run.label}}</h2>`
    + `<p>Public sampler median: ${{publicLatency}} ms; `
    + `profiled-call median: ${{profiledLatency}} ms; `
    + `sum of phase medians: ${{total.toFixed(3)}} ms</p>`;
  const bar = document.createElement('div'); bar.className = 'bar';
  const legend = document.createElement('div'); legend.className = 'legend';
  Object.entries(run.phases_ms).forEach(([name, ms], index) => {{
    const color = colors[index % colors.length];
    const width = 100 * ms / total;
    const part = document.createElement('div'); part.className = 'phase';
    part.style.width = `${{width}}%`; part.style.background = color;
    part.title = `${{name}}: ${{ms.toFixed(3)}} ms`;
    part.textContent = `${{name}}`;
    bar.appendChild(part);
    const key = document.createElement('span');
    key.innerHTML = `<i class="swatch" style="background:${{color}}"></i>`
      + `${{name}}: ${{ms.toFixed(3)}} ms`;
    legend.appendChild(key);
  }}); item.append(bar, legend); root.appendChild(item);
}});</script></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema in '{path}'")
    return data


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _add_run_parser(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser("run", help="run one sampler mode")
    parser.add_argument("--mode", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--split", default="test", choices=("train", "val", "test")
    )
    parser.add_argument(
        "--batch-size", type=parse_positive_ints, default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--fanout", type=parse_fanout, action="append")
    parser.add_argument("--include-exhaustive", action="store_true")
    parser.add_argument("--parity-batch-size", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeline-html", type=Path)
    parser.add_argument("--timeline-trace", type=Path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    compare = subparsers.add_parser(
        "compare", help="compare CPU and CUDA JSON"
    )
    compare.add_argument("--cpu", type=Path, required=True)
    compare.add_argument("--cuda", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the benchmark CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        result = compare_results(_load_json(args.cpu), _load_json(args.cuda))
        _write_json(args.output, result)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return

    if args.warmups < 0 or args.repetitions <= 0:
        raise ValueError(
            "Expected non-negative warmups and positive repetitions"
        )
    if args.parity_batch_size <= 0:
        raise ValueError("Expected a positive exhaustive parity batch size")
    if (args.timeline_html is None) != (args.timeline_trace is None):
        raise ValueError(
            "Provide both --timeline-html and --timeline-trace, or neither"
        )
    result = run_benchmark(args)
    _write_json(args.output, result)
    if args.timeline_html is not None:
        write_timeline_artifacts(
            result, args.timeline_html, args.timeline_trace
        )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
