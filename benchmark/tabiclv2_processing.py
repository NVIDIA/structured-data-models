"""Benchmark TabICLv2 Processor and Recipe overhead.

Dataset construction, correctness checks, and fresh-processor preparation are
outside the timed region. The full matrix is the Cartesian product of task,
size, and six binary data characteristics; detailed stage measurements use
one-factor-at-a-time cases plus two combined stress cases.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models.base import Model
from sdm.models.tabiclv2.recipe import default_recipe
from sdm.processing import (
    CategoricalAlign,
    ClassShuffle,
    FeaturePermute,
    HardClip,
    InvertibleMixin,
    Power,
    Processor,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
)
from torch import Tensor

Task = Literal["classification", "regression"]
LOGGER = logging.getLogger(__name__)

SIZES = {
    "tiny": (300, 10),
    "small": (1_000, 50),
    "medium": (10_000, 100),
    "large": (50_000, 100),
}

FACTOR_NAMES = (
    "constant",
    "hard_outlier",
    "sigma_outlier",
    "categorical",
    "missing",
    "unknown_category",
)


@dataclass(frozen=True)
class Characteristics:
    """Binary characteristics used to construct one synthetic workload."""

    constant: bool = False
    hard_outlier: bool = False
    sigma_outlier: bool = False
    categorical: bool = False
    missing: bool = False
    unknown_category: bool = False

    @property
    def label(self) -> str:
        """Return a stable label for the active factors."""
        active = [name for name in FACTOR_NAMES if getattr(self, name)]
        return "+".join(active) if active else "baseline"


@dataclass(frozen=True)
class Workload:
    """Tensorized benchmark data and its descriptive metadata."""

    name: str
    task: Task
    rows: int
    features: int
    train_rows: int
    x: TableTensor
    y: TableTensor
    characteristics: Characteristics


@dataclass(frozen=True)
class BenchmarkResult:
    """One operation's timing distribution and workload metadata."""

    operation: str
    task_type: Task
    processor_or_recipe: str
    size: str
    row_count: int
    feature_count: int
    train_row_count: int
    dataset_characteristics: str
    characteristics: dict[str, bool]
    median_ms: float
    p95_ms: float
    standard_deviation_ms: float
    peak_memory_bytes: int | None
    device: str
    gpu_model: str | None
    dtype: str
    repetitions: int
    correctness_status: str


class _ZeroModel(Model):
    """A deterministic zero-cost head used to isolate recipe overhead."""

    supports_related_tables = False

    @classmethod
    def default_recipe(cls):
        return default_recipe()

    def _forward(self, x, y, related_tables, cache):
        del related_tables, cache
        n_test = x.size(-2) - y.size(-1)
        width = 999 if y.is_floating_point() else 10
        return x.new_zeros((*x.shape[:-2], n_test, width))


def build_workload(
    *,
    size: str,
    task: Task,
    characteristics: Characteristics,
    seed: int = 0,
) -> Workload:
    """Build deterministic train/query data outside timed regions."""
    rows, features = SIZES[size]
    train_rows = max(2, int(rows * 0.8))
    generator = torch.Generator().manual_seed(seed)
    use_categorical = (
        characteristics.categorical or characteristics.unknown_category
    )
    categorical_features = max(1, features // 10) if use_categorical else 0
    numerical_features = features - categorical_features

    numerical = torch.randn(
        rows,
        numerical_features,
        generator=generator,
        dtype=torch.float32,
    )
    if characteristics.constant and numerical_features > 0:
        numerical[:, 0] = 1.0
    if characteristics.missing and numerical_features > 0:
        count = max(1, numerical.numel() // 100)
        flat = torch.randperm(numerical.numel(), generator=generator)[:count]
        numerical.reshape(-1)[flat] = torch.nan
    if characteristics.hard_outlier and numerical_features > 0:
        numerical[train_rows:, -1] = 1e12
    if characteristics.sigma_outlier and numerical_features > 0:
        column = min(1, numerical_features - 1)
        numerical[train_rows:, column] = 20.0

    blocks: dict[str, Tensor] = {"numerical": numerical}
    columns: dict[str, tuple[str, ...]] = {
        "numerical": tuple(f"num_{i}" for i in range(numerical_features))
    }
    if categorical_features > 0:
        vocabulary_size = 16
        codes = torch.randint(
            vocabulary_size,
            (rows, categorical_features),
            generator=generator,
            dtype=torch.int32,
        )
        if characteristics.missing:
            codes[::97] = -1
        categories: list[Tensor] = []
        for index in range(categorical_features):
            category = torch.arange(vocabulary_size + 1)
            categories.append(category)
            if characteristics.unknown_category:
                codes[train_rows:, index] = vocabulary_size
        blocks["categorical"] = CategoricalTensor(
            data=codes,
            categories=tuple(categories),
        )
        columns["categorical"] = tuple(
            f"cat_{i}" for i in range(categorical_features)
        )

    x = TableTensor(columns=columns, **blocks)
    if task == "classification":
        target_codes = torch.arange(train_rows, dtype=torch.int64) % 2
        y = TableTensor(
            columns={"categorical": ("target",)},
            categorical=CategoricalTensor(
                data=target_codes.unsqueeze(-1),
                categories=(StringTensor.from_list(["negative", "positive"]),),
            ),
        )
    else:
        target = torch.linspace(-3.0, 3.0, train_rows).unsqueeze(-1)
        y = TableTensor.from_tensor(target, columns=("target",))

    return Workload(
        name=size,
        task=task,
        rows=rows,
        features=features,
        train_rows=train_rows,
        x=x,
        y=y,
        characteristics=characteristics,
    )


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return ordered[index]


def _measure(
    prepare: Callable[[], Callable[[], Any]],
    *,
    repetitions: int,
    warmups: int = 1,
) -> tuple[float, float, float, int | None]:
    for _ in range(warmups):
        prepare()()

    durations: list[float] = []
    peak_memory: int | None = None
    for _ in range(repetitions):
        operation = prepare()
        started = time.perf_counter_ns()
        operation()
        durations.append((time.perf_counter_ns() - started) / 1e6)

    return (
        statistics.median(durations),
        _percentile_95(durations),
        statistics.pstdev(durations),
        peak_memory,
    )


def _result(
    workload: Workload,
    *,
    operation: str,
    owner: str,
    repetitions: int,
    timing: tuple[float, float, float, int | None],
) -> BenchmarkResult:
    median, p95, deviation, peak = timing
    return BenchmarkResult(
        operation=operation,
        task_type=workload.task,
        processor_or_recipe=owner,
        size=workload.name,
        row_count=workload.rows,
        feature_count=workload.features,
        train_row_count=workload.train_rows,
        dataset_characteristics=workload.characteristics.label,
        characteristics=asdict(workload.characteristics),
        median_ms=median,
        p95_ms=p95,
        standard_deviation_ms=deviation,
        peak_memory_bytes=peak,
        device="cpu",
        gpu_model=None,
        dtype="torch.float32",
        repetitions=repetitions,
        correctness_status="pass",
    )


def _classification_columns(target: TableTensor) -> tuple[str, ...]:
    values = target.categorical.categories[0].tolist()
    return tuple(str(value) for value in values)


def _mapping_setup(workload: Workload):
    recipe = default_recipe()
    torch.manual_seed(0)
    transformed = recipe.target.fit_transform(workload.y)
    if workload.task == "classification":
        target = transformed.categorical.as_tensor().squeeze(-1)
        member = _classification_columns(transformed)
        canonical = _classification_columns(workload.y)
        raw = torch.zeros(workload.rows - workload.train_rows, 10)
    else:
        target = transformed.numerical.squeeze(-1)
        raw = torch.zeros(workload.rows - workload.train_rows, 999)
        member = None
        canonical = None
    return recipe, raw, target, member, canonical


def _assert_workload_correct(workload: Workload) -> None:
    recipe = default_recipe()
    torch.manual_seed(0)
    context = workload.x[: workload.train_rows]
    recipe.features.fit(context)
    features = recipe.features.transform(workload.x)
    target = recipe.target.fit_transform(workload.y)
    assert features.shape[0] == workload.rows
    assert features.numerical.shape[-1] <= workload.features
    assert target.shape == workload.y.shape
    if workload.characteristics.unknown_category:
        # Unknown codes are aligned to -1 before the later numerical stages.
        align = next(
            module
            for module in recipe.features.modules()
            if isinstance(module, CategoricalAlign)
        )
        aligned = align.transform(workload.x.select_stypes(Stype.categorical))
        assert bool((aligned.categorical[workload.train_rows :] == -1).any())

    model = _ZeroModel()
    torch.manual_seed(0)
    output = model(workload.x, workload.y, recipe=default_recipe())
    expected_width = 2 if workload.task == "classification" else 999
    assert output.shape == (
        workload.rows - workload.train_rows,
        expected_width,
    )
    assert torch.isfinite(output).all()


def benchmark_pipeline(
    workload: Workload,
    *,
    repetitions: int,
    detailed: bool,
) -> list[BenchmarkResult]:
    """Benchmark logical stages and total Recipe overhead."""
    _assert_workload_correct(workload)
    results: list[BenchmarkResult] = []
    context = workload.x[: workload.train_rows]

    def add(operation, prepare):
        timing = _measure(prepare, repetitions=repetitions)
        results.append(
            _result(
                workload,
                operation=operation,
                owner="TabICLv2 Recipe",
                repetitions=repetitions,
                timing=timing,
            )
        )

    if detailed:

        def feature_fit_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            return lambda: recipe.features.fit(context)

        add(
            "feature_fit",
            feature_fit_prepare,
        )

        def feature_transform_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            recipe.features.fit(context)
            return lambda: recipe.features.transform(workload.x)

        add("feature_transform", feature_transform_prepare)

        def feature_fit_transform_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            return lambda: recipe.features.fit_transform(context)

        add(
            "feature_fit_transform",
            feature_fit_transform_prepare,
        )

        def target_fit_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            return lambda: recipe.target.fit(workload.y)

        add(
            "target_fit",
            target_fit_prepare,
        )

        def target_transform_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            recipe.target.fit(workload.y)
            return lambda: recipe.target.transform(workload.y)

        add("target_transform", target_transform_prepare)

        def target_fit_transform_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)
            return lambda: recipe.target.fit_transform(workload.y)

        add(
            "target_fit_transform",
            target_fit_transform_prepare,
        )

        def preprocessing_prepare():
            recipe = default_recipe()
            torch.manual_seed(0)

            def operation():
                recipe.features.fit(context)
                recipe.target.fit_transform(workload.y)
                recipe.features.transform(workload.x)

            return operation

        add("preprocessing_total", preprocessing_prepare)

        if workload.task == "regression":

            def inverse_prepare():
                recipe, raw, _, _, _ = _mapping_setup(workload)
                inverse = cast(InvertibleMixin, recipe.target)
                table = TableTensor.from_tensor(raw)
                return lambda: inverse.inverse_transform(table)

            add("target_inverse_transform", inverse_prepare)

        def mapping_prepare():
            recipe, raw, target, member, canonical = _mapping_setup(workload)
            model = _ZeroModel()
            return lambda: model._postprocess(
                raw,
                target,
                recipe,
                member_columns=member,
                canonical_columns=canonical,
            )

        add("model_output_inverse_mapping", mapping_prepare)

        def output_prepare():
            recipe, raw, target, member, canonical = _mapping_setup(workload)
            model = _ZeroModel()
            mapped = model._postprocess(
                raw,
                target,
                recipe,
                member_columns=member,
                canonical_columns=canonical,
            )
            table = TableTensor.from_tensor(mapped)
            return lambda: recipe.output.transform(table)

        add("output_transform", output_prepare)

    def total_prepare():
        model = _ZeroModel()
        recipe = default_recipe()
        torch.manual_seed(0)
        return lambda: model(workload.x, workload.y, recipe=recipe)

    add("total_recipe_overhead", total_prepare)
    return results


def _processor_inputs(
    workload: Workload,
) -> list[tuple[str, Processor, TableTensor]]:
    numeric = workload.x.select_stypes(Stype.numerical)
    inputs: list[tuple[str, Processor, TableTensor]] = [
        ("StandardScale", StandardScale(epsilon=1e-6), numeric),
        ("HardClip", HardClip(min_value=-100, max_value=100), numeric),
        ("Power", Power(), numeric),
        ("SigmaClip", SigmaClip(threshold=4), numeric),
        ("FeaturePermute", FeaturePermute(method="shift"), numeric),
        (
            "SoftmaxTemperature",
            SoftmaxTemperature(temperature=0.9),
            TableTensor.from_tensor(torch.zeros(workload.rows, 10)),
        ),
    ]
    categorical = workload.x.select_stypes(Stype.categorical)
    if categorical.size(-1) > 0:
        inputs.append(
            (
                "CategoricalAlign",
                CategoricalAlign(order="sorted"),
                categorical,
            )
        )
    if workload.task == "classification":
        inputs.append(
            ("ClassShuffle", ClassShuffle(method="shift"), workload.y)
        )
    return inputs


def benchmark_processors(
    workload: Workload,
    *,
    repetitions: int,
) -> list[BenchmarkResult]:
    """Benchmark supported operations for individual Processors."""
    results: list[BenchmarkResult] = []
    for name, template, table in _processor_inputs(workload):
        fit_table = table[: min(workload.train_rows, table.size(-2))]

        torch.manual_seed(0)
        fitted = deepcopy(template).fit(fit_table)
        transformed = fitted.transform(table)
        assert transformed.size(-2) == table.size(-2)

        operations: list[tuple[str, Callable[[], Callable[[], Any]]]] = []
        if template.requires_fit:

            def fit_prepare(template=template, fit_table=fit_table):
                processor = deepcopy(template)
                torch.manual_seed(0)
                return lambda: processor.fit(fit_table)

            def fit_transform_prepare(template=template, fit_table=fit_table):
                processor = deepcopy(template)
                torch.manual_seed(0)
                return lambda: processor.fit_transform(fit_table)

            operations.extend(
                [
                    ("fit", fit_prepare),
                    ("fit_transform", fit_transform_prepare),
                ]
            )

        def transform_prepare(fitted=fitted, table=table):
            processor = deepcopy(fitted)
            return lambda: processor.transform(table)

        operations.append(("transform", transform_prepare))
        if isinstance(template, InvertibleMixin):
            inverse_input = fitted.transform(table)

            def inverse_prepare(fitted=fitted, output=inverse_input):
                processor = deepcopy(fitted)
                return lambda: processor.inverse_transform(output)

            operations.append(("inverse_transform", inverse_prepare))

        for operation, prepare in operations:
            timing = _measure(prepare, repetitions=repetitions)
            results.append(
                _result(
                    workload,
                    operation=operation,
                    owner=name,
                    repetitions=repetitions,
                    timing=timing,
                )
            )
    return results


def detailed_characteristics() -> tuple[Characteristics, ...]:
    """Return baseline, one-factor, and combined stress cases."""
    baseline = Characteristics()
    single = tuple(Characteristics(**{name: True}) for name in FACTOR_NAMES)
    return (
        baseline,
        *single,
        Characteristics(
            constant=True,
            hard_outlier=True,
            categorical=True,
            missing=True,
            unknown_category=True,
        ),
        Characteristics(**dict.fromkeys(FACTOR_NAMES, True)),
    )


def full_characteristics() -> Iterable[Characteristics]:
    """Yield the Cartesian product of all binary characteristics."""
    for values in itertools.product((False, True), repeat=len(FACTOR_NAMES)):
        yield Characteristics(**dict(zip(FACTOR_NAMES, values, strict=True)))


def run(
    *,
    sizes: Iterable[str],
    matrix: Literal["smoke", "oat", "full"],
    repetitions: int,
    include_processors: bool,
) -> list[BenchmarkResult]:
    """Run the selected matrix and return machine-readable results."""
    results: list[BenchmarkResult] = []
    for size in sizes:
        for task in cast(tuple[Task, Task], ("classification", "regression")):
            LOGGER.info(
                "benchmarking size=%s task=%s matrix=%s",
                size,
                task,
                matrix,
            )
            characteristics = (
                (Characteristics(),)
                if matrix == "smoke"
                else detailed_characteristics()
            )
            for values in characteristics:
                workload = build_workload(
                    size=size,
                    task=task,
                    characteristics=values,
                )
                results.extend(
                    benchmark_pipeline(
                        workload,
                        repetitions=repetitions,
                        detailed=True,
                    )
                )
                if include_processors and values.label in {
                    "baseline",
                    "categorical",
                    "hard_outlier",
                    "sigma_outlier",
                }:
                    results.extend(
                        benchmark_processors(
                            workload,
                            repetitions=repetitions,
                        )
                    )

            if matrix == "full":
                detailed = {value.label for value in characteristics}
                for values in full_characteristics():
                    if values.label in detailed:
                        continue
                    workload = build_workload(
                        size=size,
                        task=task,
                        characteristics=values,
                    )
                    results.extend(
                        benchmark_pipeline(
                            workload,
                            repetitions=repetitions,
                            detailed=False,
                        )
                    )
    return results


def write_results(results: list[BenchmarkResult], output: Path) -> None:
    """Write nested JSON and flattened CSV benchmark artifacts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    payload = {
        "schema_version": 1,
        "reference_commit": "f719c886a586ed4a29236345e319ac1ea596c478",
        "sizes": SIZES,
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_model": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "results": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = output.with_suffix(".csv")
    flat_rows = []
    for row in rows:
        row = dict(row)
        characteristics = row.pop("characteristics")
        row.update(characteristics)
        flat_rows.append(row)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(flat_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=tuple(SIZES),
        default=tuple(SIZES),
    )
    parser.add_argument(
        "--matrix",
        choices=("smoke", "oat", "full"),
        default="smoke",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--include-processors", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/tabiclv2_processing.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    results = run(
        sizes=args.sizes,
        matrix=args.matrix,
        repetitions=args.repetitions,
        include_processors=args.include_processors,
    )
    write_results(results, args.output)
    LOGGER.info("wrote %d results to %s", len(results), args.output)


if __name__ == "__main__":
    main()
