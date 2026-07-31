"""Benchmark ensemble-aware TabICLv2 processing and model scheduling.

Dataset construction, correctness checks, and host/device transfers are kept
outside processing timings. CUDA measurements use warm-up runs, events, and an
explicit synchronization after every measured operation.
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
from typing import Any, Literal, cast

import torch

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel, TabICLv2
from sdm.models.tabiclv2.recipe import default_recipe
from sdm.processing import (
    AlignCategories,
    Clip,
    ClipSigma,
    DropConstantColumns,
    ImputeMean,
    PowerTransform,
    Recipe,
    Sequential,
    ShuffleColumns,
    Standardize,
)

Task = Literal["classification", "regression"]
Mode = Literal["parallel", "sequential"]


@dataclass(frozen=True)
class Workload:
    """Prepared context/query tables and descriptive metadata."""

    task: Task
    x_context: TableTensor
    target: TableTensor
    x_query: TableTensor
    rows: int
    context_rows: int
    query_rows: int
    features: int

    @property
    def device(self) -> torch.device:
        """Return the common table device."""
        return self.x_context.device

    def to(self, device: torch.device | str) -> Workload:
        """Copy all workload tables to ``device``."""
        return self.__class__(
            task=self.task,
            x_context=self.x_context.to(device),
            target=self.target.to(device),
            x_query=self.x_query.to(device),
            rows=self.rows,
            context_rows=self.context_rows,
            query_rows=self.query_rows,
            features=self.features,
        )


@dataclass(frozen=True)
class Measurement:
    """One synchronized operation timing and its execution metadata."""

    operation: str
    task: Task
    device: str
    mode: str | None
    rows: int
    context_rows: int
    query_rows: int
    features: int
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
    throughput_rows_per_second: float


class _ProcessingBoundaryModel(ICLModel):
    """Zero-compute core that retains the complete Recipe boundary."""

    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables = False

    @classmethod
    def default_recipe(cls) -> Recipe:
        """Return the production TabICLv2 Recipe."""
        return default_recipe()

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        del (
            x_context,
            related_context_tables,
            related_query_tables,
            cache,
            generator,
            kwargs,
        )
        assert x_query is not None
        assert y_context is not None
        if y_context.categorical.size(-1) > 0:
            categories = y_context.categorical.categories[0]
            width = categories.numel()
            columns = tuple(str(value) for value in categories.tolist())
        else:
            width = 999
            columns = tuple(f"q{index:03d}" for index in range(1, 1000))
        numerical = x_query.numerical.new_zeros((*x_query.size()[:-1], width))
        return TableTensor(
            columns={Stype.numerical: columns},
            numerical=numerical,
        )


def build_workload(
    *,
    task: Task,
    context_rows: int = 40_000,
    query_rows: int = 10_000,
    features: int = 100,
    categorical_features: int = 10,
    vocabulary_size: int = 4_096,
    seed: int = 7,
) -> Workload:
    """Build the deterministic mixed 50k workload used by PR271.

    The data contains numerical and categorical missing values, one constant
    feature, hard and sigma outliers, high-cardinality categories, and a
    query-only category.
    """
    if context_rows < 2 or query_rows < 1:
        raise ValueError("Expected at least two context and one query row.")
    if not 0 <= categorical_features < features:
        raise ValueError(
            "'categorical_features' must be non-negative and below 'features'."
        )
    if vocabulary_size < 2:
        raise ValueError("'vocabulary_size' must be at least two.")

    numerical_features = features - categorical_features
    generator = torch.Generator().manual_seed(seed)

    def feature_table(rows: int, *, query: bool) -> TableTensor:
        numerical = torch.randn(
            rows,
            numerical_features,
            generator=generator,
            dtype=torch.float32,
        )
        numerical[::101, 0] = torch.nan
        numerical[:, 1] = 1.0
        if query:
            numerical[-1, 2] = 1e12
            numerical[-2, 3] = 20.0

        columns: dict[Stype, tuple[str, ...]] = {
            Stype.numerical: tuple(
                f"num_{index}" for index in range(numerical_features)
            )
        }
        if categorical_features == 0:
            return TableTensor(columns=columns, numerical=numerical)

        code = torch.randint(
            vocabulary_size,
            (rows, categorical_features),
            generator=generator,
            dtype=torch.int32,
        )
        code[::97] = -1
        category_count = vocabulary_size + int(query)
        if query:
            code[-1] = vocabulary_size
        columns[Stype.categorical] = tuple(
            f"cat_{index}" for index in range(categorical_features)
        )
        return TableTensor(
            columns=columns,
            numerical=numerical,
            categorical=CategoricalTensor(
                code=code,
                categories=tuple(
                    torch.arange(category_count)
                    for _ in range(categorical_features)
                ),
            ),
        )

    if task == "classification":
        target = TableTensor(
            columns={Stype.categorical: ("target",)},
            categorical=CategoricalTensor(
                code=(
                    torch.arange(context_rows, dtype=torch.int32) % 3
                ).unsqueeze(-1),
                categories=(torch.tensor([10, 20, 30]),),
            ),
        )
    else:
        values = torch.linspace(-3.0, 3.0, context_rows).unsqueeze(-1)
        target = TableTensor.from_tensor(values, columns=("target",))

    return Workload(
        task=task,
        x_context=feature_table(context_rows, query=False),
        target=target,
        x_query=feature_table(query_rows, query=True),
        rows=context_rows + query_rows,
        context_rows=context_rows,
        query_rows=query_rows,
        features=features,
    )


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


def _cpu_peak(prepare: Callable[[], Callable[[], Any]]) -> tuple[int, int]:
    operation = prepare()
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


def measure(
    prepare: Callable[[], Callable[[], Any]],
    *,
    operation: str,
    workload: Workload,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    mode: str | None = None,
    profile_kernels: bool = False,
) -> Measurement:
    """Measure one prepared operation with synchronized CPU/CUDA timing."""
    device = workload.device
    for _ in range(warmups):
        result = prepare()()
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
        run = prepare()
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
        result = run()
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
            current_peak = torch.cuda.max_memory_allocated(device)
            peak_delta = max(peak_delta, current_peak - baseline)
            absolute_peak = max(absolute_peak, current_peak)
        else:
            finished = time.perf_counter_ns()
        wall.append((finished - started) / 1e6)
        del result

    if device.type == "cpu":
        peak_delta, absolute_peak = _cpu_peak(prepare)

    kernel_time: float | None = None
    if device.type == "cuda" and profile_kernels:
        run = prepare()
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            result = run()
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
    cuda_event_median = (
        statistics.median(cuda_elapsed) if cuda_elapsed else None
    )
    return Measurement(
        operation=operation,
        task=workload.task,
        device=str(device),
        mode=mode,
        rows=workload.rows,
        context_rows=workload.context_rows,
        query_rows=workload.query_rows,
        features=workload.features,
        num_estimators=num_estimators,
        dtype=str(workload.x_context.dtype),
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
        cuda_event_elapsed_median_ms=cuda_event_median,
        cuda_kernel_time_ms=kernel_time,
        peak_memory_bytes=peak_delta,
        absolute_peak_memory_bytes=absolute_peak,
        throughput_rows_per_second=workload.rows / (median / 1000),
    )


def _raw_outputs(
    workload: Workload,
    *,
    num_estimators: int,
) -> tuple[TableTensor, ...]:
    if workload.task == "classification":
        columns = ("10", "20", "30")
        width = 3
    else:
        columns = tuple(f"q{index:03d}" for index in range(1, 1000))
        width = 999
    output = TableTensor(
        columns={Stype.numerical: columns},
        numerical=torch.zeros(
            workload.query_rows,
            width,
            device=workload.device,
        ),
    )
    return (output,) * num_estimators


def _assert_correct(workload: Workload, num_estimators: int) -> None:
    recipe = default_recipe()
    context, target, _ = recipe.fit_transform(
        workload.x_context,
        workload.target,
        num_members=num_estimators,
        generator=torch.Generator().manual_seed(42),
    )
    query, _ = recipe.transform(workload.x_query)
    output = recipe.transform_output(
        _raw_outputs(workload, num_estimators=num_estimators)
    )
    assert context.num_members == target.num_members == num_estimators
    assert query.num_members == num_estimators
    assert output.size(-2) == workload.query_rows
    assert output.numerical.isfinite().all()


def benchmark_recipe(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    modes: Sequence[Mode],
    profile_kernels: bool,
) -> list[Measurement]:
    """Measure Recipe stages and the zero-compute processing boundary."""
    _assert_correct(workload, num_estimators)
    results: list[Measurement] = []

    def add(operation: str, prepare: Callable[[], Callable[[], Any]]) -> None:
        results.append(
            measure(
                prepare,
                operation=operation,
                workload=workload,
                num_estimators=num_estimators,
                repetitions=repetitions,
                warmups=warmups,
                profile_kernels=profile_kernels,
            )
        )

    def fit_prepare() -> Callable[[], Any]:
        recipe = default_recipe()
        return lambda: recipe.fit_transform(
            workload.x_context,
            workload.target,
            num_members=num_estimators,
            generator=torch.Generator().manual_seed(42),
        )

    add("recipe_fit_transform", fit_prepare)

    def transform_prepare() -> Callable[[], Any]:
        recipe = default_recipe()
        recipe.fit_transform(
            workload.x_context,
            workload.target,
            num_members=num_estimators,
            generator=torch.Generator().manual_seed(42),
        )
        return lambda: recipe.transform(workload.x_query)

    add("recipe_query_transform", transform_prepare)

    raw_outputs = _raw_outputs(workload, num_estimators=num_estimators)

    def output_prepare() -> Callable[[], Any]:
        recipe = default_recipe()
        recipe.fit_transform(
            workload.x_context,
            workload.target,
            num_members=num_estimators,
            generator=torch.Generator().manual_seed(42),
        )
        return lambda: recipe.transform_output(raw_outputs)

    add("recipe_output_transform", output_prepare)

    def total_prepare() -> Callable[[], Any]:
        recipe = default_recipe()

        def run() -> TableTensor:
            recipe.fit_transform(
                workload.x_context,
                workload.target,
                num_members=num_estimators,
                generator=torch.Generator().manual_seed(42),
            )
            recipe.transform(workload.x_query)
            return recipe.transform_output(raw_outputs)

        return run

    add("recipe_total", total_prepare)

    boundary = _ProcessingBoundaryModel().eval()
    for mode in modes:

        def boundary_prepare(mode: Mode = mode) -> Callable[[], Any]:
            return lambda: boundary(
                workload.x_context,
                workload.target,
                workload.x_query,
                num_estimators=num_estimators,
                ensemble_mode=mode,
                generator=torch.Generator().manual_seed(42),
            )

        results.append(
            measure(
                boundary_prepare,
                operation="zero_core_processing_boundary",
                workload=workload,
                num_estimators=num_estimators,
                repetitions=repetitions,
                warmups=warmups,
                mode=mode,
                profile_kernels=profile_kernels,
            )
        )
    return results


def _prepared_numerical_inputs(
    workload: Workload,
) -> tuple[TableTensor, TableTensor, TableTensor]:
    context = workload.x_context.select_stypes(Stype.numerical)
    query = workload.x_query.select_stypes(Stype.numerical)
    prefix = Sequential(
        ImputeMean(),
        DropConstantColumns(),
        Standardize(epsilon=1e-6),
        Clip(min_value=-100.0, max_value=100.0),
    )
    fitted_context = prefix.fit_transform(context)
    fitted_query = prefix.transform(query)
    return context, fitted_context, fitted_query


def benchmark_processors(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    profile_kernels: bool,
) -> list[Measurement]:
    """Measure the dominant scalar Processor operations on Recipe inputs."""
    raw, power_context, power_query = _prepared_numerical_inputs(workload)
    categorical_context = workload.x_context.select_stypes(Stype.categorical)
    categorical_query = workload.x_query.select_stypes(Stype.categorical)

    cases: list[tuple[str, Callable[[], Callable[[], Any]]]] = []

    def impute_prepare() -> Callable[[], Any]:
        processor = ImputeMean()
        return lambda: processor.fit_transform(raw)

    cases.append(("ImputeMean.fit_transform", impute_prepare))

    imputed = ImputeMean().fit_transform(raw)

    def drop_prepare() -> Callable[[], Any]:
        processor = DropConstantColumns()
        return lambda: processor.fit_transform(imputed)

    cases.append(("DropConstantColumns.fit_transform", drop_prepare))
    dropped = DropConstantColumns().fit_transform(imputed)

    def standardize_prepare() -> Callable[[], Any]:
        processor = Standardize(epsilon=1e-6)
        return lambda: processor.fit_transform(dropped)

    cases.append(("Standardize.fit_transform", standardize_prepare))

    def power_fit_prepare() -> Callable[[], Any]:
        processor = PowerTransform()
        return lambda: processor.fit(power_context)

    cases.append(("PowerTransform.fit", power_fit_prepare))

    def power_fit_transform_prepare() -> Callable[[], Any]:
        processor = PowerTransform()
        return lambda: processor.fit_transform(power_context)

    cases.append(("PowerTransform.fit_transform", power_fit_transform_prepare))
    fitted_power = PowerTransform().fit(power_context)

    def power_transform_prepare() -> Callable[[], Any]:
        return lambda: fitted_power.transform(power_query)

    cases.append(("PowerTransform.transform", power_transform_prepare))
    power_output = fitted_power.transform(power_context)

    def power_inverse_prepare() -> Callable[[], Any]:
        return lambda: fitted_power.inverse_transform(power_output)

    cases.append(("PowerTransform.inverse_transform", power_inverse_prepare))

    def sigma_prepare() -> Callable[[], Any]:
        processor = ClipSigma(threshold=4.0)
        return lambda: processor.fit_transform(power_output)

    cases.append(("ClipSigma.fit_transform", sigma_prepare))

    def shuffle_prepare() -> Callable[[], Any]:
        processor = ShuffleColumns(method="shift")
        return lambda: processor.fit_transform(power_output)

    cases.append(("ShuffleColumns.fit_transform", shuffle_prepare))

    if categorical_context.size(-1) > 0:

        def align_fit_prepare() -> Callable[[], Any]:
            processor = AlignCategories(sort_by="value")
            return lambda: processor.fit_transform(categorical_context)

        cases.append(("AlignCategories.fit_transform", align_fit_prepare))
        fitted_align = AlignCategories(sort_by="value").fit(
            categorical_context
        )

        def align_transform_prepare() -> Callable[[], Any]:
            return lambda: fitted_align.transform(categorical_query)

        cases.append(("AlignCategories.transform", align_transform_prepare))

    return [
        measure(
            prepare,
            operation=operation,
            workload=workload,
            num_estimators=num_estimators,
            repetitions=repetitions,
            warmups=warmups,
            profile_kernels=profile_kernels,
        )
        for operation, prepare in cases
    ]


def benchmark_transfers(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    profile_kernels: bool,
) -> list[Measurement]:
    """Measure host/device transfers separately from processing operations."""
    if workload.device.type != "cuda":
        return []
    cpu = workload.to("cpu")

    def h2d_prepare() -> Callable[[], Any]:
        return lambda: (
            cpu.x_context.to(workload.device),
            cpu.target.to(workload.device),
            cpu.x_query.to(workload.device),
        )

    def d2h_prepare() -> Callable[[], Any]:
        return lambda: (
            workload.x_context.cpu(),
            workload.target.cpu(),
            workload.x_query.cpu(),
        )

    return [
        measure(
            h2d_prepare,
            operation="host_to_device_transfer",
            workload=workload,
            num_estimators=num_estimators,
            repetitions=repetitions,
            warmups=warmups,
            profile_kernels=profile_kernels,
        ),
        measure(
            d2h_prepare,
            operation="device_to_host_transfer",
            workload=workload,
            num_estimators=num_estimators,
            repetitions=repetitions,
            warmups=warmups,
            profile_kernels=profile_kernels,
        ),
    ]


def benchmark_actual_model(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    modes: Sequence[Mode],
    profile_kernels: bool,
) -> list[Measurement]:
    """Measure the random-weight production architecture end to end."""
    model = TabICLv2(pretrained=False, device=workload.device).eval()
    results = []
    for mode in modes:

        def prepare(mode: Mode = mode) -> Callable[[], Any]:
            return lambda: model(
                workload.x_context,
                workload.target,
                workload.x_query,
                num_estimators=num_estimators,
                ensemble_mode=mode,
                generator=torch.Generator().manual_seed(42),
            )

        results.append(
            measure(
                prepare,
                operation="actual_model_end_to_end",
                workload=workload,
                num_estimators=num_estimators,
                repetitions=repetitions,
                warmups=warmups,
                mode=mode,
                profile_kernels=profile_kernels,
            )
        )
    return results


def metadata() -> dict[str, object]:
    """Return stable hardware and software metadata for the result file."""
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


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the requested benchmark matrix."""
    results: list[Measurement] = []
    modes = cast(tuple[Mode, ...], tuple(args.modes))
    for task in cast(tuple[Task, ...], tuple(args.tasks)):
        cpu_workload = build_workload(
            task=task,
            context_rows=args.context_rows,
            query_rows=args.query_rows,
            features=args.features,
            categorical_features=args.categorical_features,
            vocabulary_size=args.vocabulary_size,
        )
        for device_name in args.devices:
            device = torch.device(device_name)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable.")
            workload = cpu_workload.to(device)
            if not args.only_model:
                results.extend(
                    benchmark_recipe(
                        workload,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        modes=modes,
                        profile_kernels=args.profile_kernels,
                    )
                )
            if (
                not args.only_model
                and args.include_processors
                and task == "classification"
            ):
                results.extend(
                    benchmark_processors(
                        workload,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        profile_kernels=args.profile_kernels,
                    )
                )
            if not args.only_model and args.include_transfers:
                results.extend(
                    benchmark_transfers(
                        workload,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        profile_kernels=args.profile_kernels,
                    )
                )
            if args.include_model:
                results.extend(
                    benchmark_actual_model(
                        workload,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        modes=modes,
                        profile_kernels=args.profile_kernels,
                    )
                )
            del workload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "metadata": metadata(),
        "configuration": {
            key: value for key, value in vars(args).items() if key != "output"
        },
        "results": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    """Parse the benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("classification", "regression"),
        default=("classification", "regression"),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=("cpu", "cuda"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("parallel", "sequential"),
        default=("parallel", "sequential"),
    )
    parser.add_argument("--context-rows", type=int, default=40_000)
    parser.add_argument("--query-rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--categorical-features", type=int, default=10)
    parser.add_argument("--vocabulary-size", type=int, default=4_096)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--include-processors", action="store_true")
    parser.add_argument("--include-transfers", action="store_true")
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--only-model", action="store_true")
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/tabiclv2_ensemble.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and persist machine-readable results."""
    args = parse_args()
    if args.only_model:
        args.include_model = True
    output = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    sys.stdout.write(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
