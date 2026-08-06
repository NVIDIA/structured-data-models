"""Revalidate ensemble-aware TabICLv2 processing.

This retains the PR #421 workload, seed, warm-up, repetition, timing, and
memory methodology while exercising the latest Recipe.bind execution path.
Dataset construction and host/device transfers remain outside timed regions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
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
from sdm.processing._recipe_execution import _RecipeExecution
from sdm.processing.ensemble import EnsembleInvertibleMixin
from sdm.tensor import EnsembleTable

Task = Literal["classification", "regression"]
Execution = Literal["vectorized", "sequential"]


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
        """Copy all workload tables to the requested device."""
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
    """One synchronized operation timing and execution metadata."""

    operation: str
    task: Task
    device: str
    execution: str | None
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
    peak_memory_bytes: int
    absolute_peak_memory_bytes: int


class _ProcessingBoundaryModel(ICLModel):
    """Zero-compute core retaining the complete production Recipe boundary."""

    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables = False

    @classmethod
    def default_recipe(cls) -> Recipe:
        """Return the production TabICLv2 recipe."""
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
        return TableTensor(
            columns={Stype.numerical: columns},
            numerical=x_query.numerical.new_zeros(
                (*x_query.size()[:-1], width)
            ),
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
    """Build the deterministic mixed 50k workload used by PR #421."""
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
        target = TableTensor.from_tensor(
            torch.linspace(-3.0, 3.0, context_rows).unsqueeze(-1),
            columns=("target",),
        )

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
    execution: Execution | None = None,
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

    return Measurement(
        operation=operation,
        task=workload.task,
        device=str(device),
        execution=execution,
        rows=workload.rows,
        context_rows=workload.context_rows,
        query_rows=workload.query_rows,
        features=workload.features,
        num_estimators=num_estimators,
        dtype=str(workload.x_context.dtype),
        repetitions=repetitions,
        warmups=warmups,
        median_ms=statistics.median(wall),
        p95_ms=_p95(wall),
        enqueue_median_ms=statistics.median(enqueue) if enqueue else None,
        synchronization_wait_median_ms=(
            statistics.median(synchronization_wait)
            if synchronization_wait
            else None
        ),
        cuda_event_elapsed_median_ms=(
            statistics.median(cuda_elapsed) if cuda_elapsed else None
        ),
        peak_memory_bytes=peak_delta,
        absolute_peak_memory_bytes=absolute_peak,
    )


def _bind(
    workload: Workload,
    num_estimators: int,
) -> _RecipeExecution:
    return default_recipe().bind(
        x_context=workload.x_context,
        y_context=workload.target,
        num_members=num_estimators,
        generator=torch.Generator().manual_seed(42),
    )


def _raw_outputs(
    workload: Workload,
    execution: _RecipeExecution,
) -> tuple[TableTensor, ...]:
    outputs = []
    for context in execution.contexts:
        if workload.task == "classification":
            categories = context.y.categorical.categories[0]
            columns = tuple(str(value) for value in categories.tolist())
            width = len(columns)
        else:
            columns = tuple(f"q{index:03d}" for index in range(1, 1000))
            width = 999
        outputs.append(
            TableTensor(
                columns={Stype.numerical: columns},
                numerical=torch.zeros(
                    workload.query_rows,
                    width,
                    device=workload.device,
                ),
            )
        )
    return tuple(outputs)


def _map_output(
    workload: Workload,
    execution: _RecipeExecution,
    outputs: Sequence[TableTensor],
) -> TableTensor:
    if workload.task == "regression" and workload.device.type == "cpu":
        mapped_members = execution.inverse_transform_target(outputs)
        return cast(
            TableTensor,
            torch.stack(
                cast(list[torch.Tensor], list(mapped_members)),
                dim=0,
            ),
        )

    stacked = cast(
        TableTensor,
        torch.stack(cast(list[torch.Tensor], list(outputs)), dim=0),
    )
    if workload.task == "classification":
        return stacked

    mapped = cast(
        EnsembleInvertibleMixin,
        execution.recipe.target,
    ).inverse_transform_ensemble(EnsembleTable._from_group(stacked))
    assert mapped.num_groups == 1
    return next(iter(mapped))


def _reduce_output(
    execution: _RecipeExecution,
    mapped: TableTensor,
) -> TableTensor:
    reduce_estimators = next(iter(execution.recipe.output))
    return reduce_estimators.transform(mapped)


def _finalize_output(
    execution: _RecipeExecution,
    reduced: TableTensor,
) -> TableTensor:
    output = reduced
    for processor in tuple(execution.recipe.output)[1:]:
        output = processor.transform(output)
    return output


def _transform_output(
    workload: Workload,
    execution: _RecipeExecution,
    outputs: Sequence[TableTensor],
) -> TableTensor:
    mapped = _map_output(workload, execution, outputs)
    reduced = _reduce_output(execution, mapped)
    return _finalize_output(execution, reduced)


def benchmark_recipe(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    executions: Sequence[Execution],
) -> list[Measurement]:
    """Measure recipe stages and the zero-core processing boundary."""
    template = _bind(workload, num_estimators)
    raw_outputs = _raw_outputs(workload, template)
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
            )
        )

    add(
        "recipe_bind",
        lambda: lambda: _bind(workload, num_estimators),
    )

    def query_prepare() -> Callable[[], Any]:
        execution = _bind(workload, num_estimators)
        return lambda: execution.transform(
            x_query=workload.x_query,
            related_query_tables=None,
        )

    add("recipe_query_transform", query_prepare)

    def output_prepare() -> Callable[[], Any]:
        execution = _bind(workload, num_estimators)
        return lambda: _transform_output(
            workload,
            execution,
            raw_outputs,
        )

    add("recipe_output_transform", output_prepare)

    def mapping_prepare() -> Callable[[], Any]:
        execution = _bind(workload, num_estimators)
        return lambda: _map_output(workload, execution, raw_outputs)

    add("canonical_output_mapping", mapping_prepare)

    mapped_output = _map_output(workload, template, raw_outputs)

    def reduction_prepare() -> Callable[[], Any]:
        execution = _bind(workload, num_estimators)
        return lambda: _reduce_output(execution, mapped_output)

    add("estimator_reduction", reduction_prepare)

    reduced_output = _reduce_output(template, mapped_output)

    def final_prepare() -> Callable[[], Any]:
        execution = _bind(workload, num_estimators)
        return lambda: _finalize_output(execution, reduced_output)

    add("final_output_processing", final_prepare)

    def total_prepare() -> Callable[[], Any]:
        def run() -> TableTensor:
            execution = _bind(workload, num_estimators)
            execution.transform(
                x_query=workload.x_query,
                related_query_tables=None,
            )
            return _transform_output(workload, execution, raw_outputs)

        return run

    add("recipe_total", total_prepare)

    boundary = _ProcessingBoundaryModel().eval()
    for execution in executions:

        def boundary_prepare(
            execution: Execution = execution,
        ) -> Callable[[], Any]:
            return lambda: boundary(
                workload.x_context,
                workload.target,
                workload.x_query,
                num_estimators=num_estimators,
                recipe_execution=execution,
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
                execution=execution,
            )
        )
    return results


def _prepared_numerical_inputs(
    workload: Workload,
) -> tuple[TableTensor, TableTensor, TableTensor]:
    raw = workload.x_context.select_stypes(Stype.numerical)
    query = workload.x_query.select_stypes(Stype.numerical)
    prefix = Sequential(
        ImputeMean(),
        DropConstantColumns(),
        Standardize(epsilon=1e-6),
        Clip(min_value=-100.0, max_value=100.0),
    )
    return raw, prefix.fit_transform(raw), prefix.transform(query)


def benchmark_processors(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
) -> list[Measurement]:
    """Measure dominant scalar Processor operations on recipe inputs."""
    raw, power_context, power_query = _prepared_numerical_inputs(workload)
    categorical_context = workload.x_context.select_stypes(Stype.categorical)
    categorical_query = workload.x_query.select_stypes(Stype.categorical)
    cases: list[tuple[str, Callable[[], Callable[[], Any]]]] = []

    cases.append(
        (
            "ImputeMean.fit_transform",
            lambda: lambda: ImputeMean().fit_transform(raw),
        )
    )
    imputed = ImputeMean().fit_transform(raw)
    cases.append(
        (
            "DropConstantColumns.fit_transform",
            lambda: lambda: DropConstantColumns().fit_transform(imputed),
        )
    )
    dropped = DropConstantColumns().fit_transform(imputed)
    cases.append(
        (
            "Standardize.fit_transform",
            lambda: lambda: Standardize(epsilon=1e-6).fit_transform(dropped),
        )
    )
    cases.append(
        (
            "PowerTransform.fit",
            lambda: lambda: PowerTransform().fit(power_context),
        )
    )
    cases.append(
        (
            "PowerTransform.fit_transform",
            lambda: lambda: PowerTransform().fit_transform(power_context),
        )
    )
    fitted_power = PowerTransform().fit(power_context)
    cases.append(
        (
            "PowerTransform.transform",
            lambda: lambda: fitted_power.transform(power_query),
        )
    )
    power_output = fitted_power.transform(power_context)
    cases.append(
        (
            "PowerTransform.inverse_transform",
            lambda: lambda: fitted_power.inverse_transform(power_output),
        )
    )
    cases.append(
        (
            "ClipSigma.fit_transform",
            lambda: (
                lambda: ClipSigma(threshold=4.0).fit_transform(power_output)
            ),
        )
    )
    cases.append(
        (
            "ShuffleColumns.fit_transform",
            lambda: (
                lambda: ShuffleColumns(method="shift").fit_transform(
                    power_output
                )
            ),
        )
    )
    if categorical_context.size(-1) > 0:
        cases.append(
            (
                "AlignCategories.fit_transform",
                lambda: (
                    lambda: AlignCategories(sort_by="value").fit_transform(
                        categorical_context
                    )
                ),
            )
        )
        fitted_align = AlignCategories(sort_by="value").fit(
            categorical_context
        )
        cases.append(
            (
                "AlignCategories.transform",
                lambda: lambda: fitted_align.transform(categorical_query),
            )
        )

    return [
        measure(
            prepare,
            operation=operation,
            workload=workload,
            num_estimators=num_estimators,
            repetitions=repetitions,
            warmups=warmups,
        )
        for operation, prepare in cases
    ]


def benchmark_actual_model(
    workload: Workload,
    *,
    num_estimators: int,
    repetitions: int,
    warmups: int,
    executions: Sequence[Execution],
) -> list[Measurement]:
    """Measure the random-weight production architecture end to end."""
    model = TabICLv2(pretrained=False, device=workload.device).eval()
    results = []
    for execution in executions:

        def prepare(
            execution: Execution = execution,
        ) -> Callable[[], Any]:
            return lambda: model(
                workload.x_context,
                workload.target,
                workload.x_query,
                num_estimators=num_estimators,
                recipe_execution=execution,
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
                execution=execution,
            )
        )
    return results


def _git_revision(ref: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", ref],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def metadata() -> dict[str, object]:
    """Return stable software, hardware, and source metadata."""
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
        "head_commit": _git_revision("HEAD"),
        "main_commit": _git_revision("origin/main"),
        "pr516_commit": "151be48af490cf550c934a126e216448f3ebc4ae",
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
    executions = cast(tuple[Execution, ...], tuple(args.executions))
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
                        executions=executions,
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
                    )
                )
            if args.include_model:
                results.extend(
                    benchmark_actual_model(
                        workload,
                        num_estimators=args.num_estimators,
                        repetitions=args.repetitions,
                        warmups=args.warmups,
                        executions=executions,
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
        "--executions",
        nargs="+",
        choices=("vectorized", "sequential"),
        default=("vectorized", "sequential"),
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
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--only-model", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/pr421_tabiclv2_revalidation.json"),
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
