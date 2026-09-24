# ruff: noqa: B023, D101, D102, D103, E501, PLC0415, T201
# Scenario closures run synchronously before their benchmark loop advances.
# This standalone harness intentionally prints and lazily imports heavy code.
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable

WARMUPS = 5
REPETITIONS = 30
BASELINE_COMMIT = "2a73246320ea4188d8b8791e136ac8f111e112d6"
REFERENCE_COMMIT = "f719c886a586ed4a29236345e319ac1ea596c478"


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def measure(
    operation: Callable[[], object],
    *,
    warmups: int = WARMUPS,
    repetitions: int = REPETITIONS,
) -> dict[str, Any]:
    wall_samples: list[float] = []
    event_samples: list[float] = []
    peak_samples: list[int] = []
    output: object | None = None

    for iteration in range(warmups + repetitions):
        output = None
        gc.collect()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start = time.perf_counter()
        start_event.record()
        output = operation()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1_000

        if iteration >= warmups:
            wall_samples.append(elapsed_ms)
            event_samples.append(start_event.elapsed_time(end_event))
            peak_samples.append(torch.cuda.max_memory_allocated() - baseline)

    del output
    gc.collect()
    return {
        "median_ms": statistics.median(wall_samples),
        "p95_ms": percentile(wall_samples, 0.95),
        "cuda_event_median_ms": statistics.median(event_samples),
        "peak_memory_bytes": max(peak_samples),
        "wall_samples_ms": wall_samples,
        "cuda_event_samples_ms": event_samples,
        "peak_memory_samples_bytes": peak_samples,
    }


def measure_interleaved(
    operations: dict[str, Callable[[], object]],
    *,
    warmup_rounds: int,
    repetitions: int,
) -> dict[str, dict[str, Any]]:
    wall_samples = {name: [] for name in operations}
    event_samples = {name: [] for name in operations}
    peak_samples = {name: [] for name in operations}
    names = tuple(operations)
    output: object | None = None

    for round_id in range(warmup_rounds + repetitions):
        order = names if round_id % 2 == 0 else tuple(reversed(names))
        for name in order:
            output = None
            gc.collect()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            start_event.record()
            output = operations[name]()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1_000
            if round_id >= warmup_rounds:
                wall_samples[name].append(elapsed_ms)
                event_samples[name].append(start_event.elapsed_time(end_event))
                peak_samples[name].append(
                    torch.cuda.max_memory_allocated() - baseline
                )

    del output
    gc.collect()
    return {
        name: {
            "median_ms": statistics.median(wall_samples[name]),
            "p95_ms": percentile(wall_samples[name], 0.95),
            "cuda_event_median_ms": statistics.median(event_samples[name]),
            "peak_memory_bytes": max(peak_samples[name]),
            "wall_samples_ms": wall_samples[name],
            "cuda_event_samples_ms": event_samples[name],
            "peak_memory_samples_bytes": peak_samples[name],
        }
        for name in names
    }


def cyclic_permutations(
    count: int, columns: int, device: torch.device
) -> Tensor:
    base = torch.arange(columns, device=device)
    offsets = torch.arange(count, device=device).remainder(columns)
    return (base.unsqueeze(0) + offsets.unsqueeze(1)).remainder(columns)


def feature_reference(
    source: Tensor,
    permutations: Tensor,
    source_ids: Sequence[int],
    permutation_ids: Sequence[int],
) -> tuple[Tensor, ...]:
    return tuple(
        source[source_id].index_select(-1, permutations[permutation_id])
        for source_id, permutation_id in zip(
            source_ids, permutation_ids, strict=True
        )
    )


def feature_coupled_cartesian(source: Tensor, permutations: Tensor) -> Tensor:
    # [K, S, R, C], with one permutation shared across S representations.
    count, columns = permutations.size()
    representations, rows, _ = source.size()
    expanded_source = source.unsqueeze(0).expand(
        count, representations, rows, columns
    )
    indices = permutations[:, None, None, :].expand(
        count, representations, rows, columns
    )
    return expanded_source.gather(-1, indices)


def feature_independent_by_source(
    source: Tensor,
    permutations: Tensor,
    source_ids: Sequence[int],
    permutation_ids: Sequence[int],
) -> tuple[Tensor, ...]:
    outputs = []
    rows = source.size(-2)
    for source_id in range(source.size(0)):
        ids = [
            permutation_id
            for member_source, permutation_id in zip(
                source_ids, permutation_ids, strict=True
            )
            if member_source == source_id
        ]
        indices = permutations[ids]
        expanded_source = (
            source[source_id]
            .unsqueeze(0)
            .expand(len(ids), rows, source.size(-1))
        )
        outputs.append(
            expanded_source.gather(
                -1,
                indices[:, None, :].expand(-1, rows, -1),
            )
        )
    return tuple(outputs)


def feature_independent_one_gather_with_source_copy(
    source: Tensor,
    permutations: Tensor,
    source_ids: Tensor,
    permutation_ids: Tensor,
) -> Tensor:
    selected_source = source.index_select(0, source_ids)
    indices = permutations.index_select(0, permutation_ids)
    return selected_source.gather(
        -1,
        indices[:, None, :].expand(-1, source.size(-2), -1),
    )


def make_feature_ensemble(
    rows: int,
    columns: int,
    estimators: int,
    *,
    representations: int,
    device: torch.device,
) -> EnsembleTable:
    names = tuple(f"f{index}" for index in range(columns))
    tables = tuple(
        TableTensor.from_tensor(
            torch.randn(rows, columns, device=device),
            columns=names,
        )
        for _ in range(representations)
    )
    return EnsembleTable.from_tables(
        tables=tables,
        member_table_ids=tuple(
            member_id % representations for member_id in range(estimators)
        ),
    )


class PlannedBatchedShuffleColumns(EnsembleProcessor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, representations: int = 2) -> None:
        super().__init__()
        self.representations = representations
        self._permutations: Tensor | None = None
        self._orders: tuple[tuple[int, ...], ...] = ()
        self._permutation_ids: tuple[int, ...] = ()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        count = math.ceil(ensemble_table.num_members / self.representations)
        columns = next(iter(ensemble_table)).numerical.size(-1)
        self._permutations = cyclic_permutations(
            count, columns, next(iter(ensemble_table)).device
        )
        self._orders = tuple(
            tuple(permutation) for permutation in self._permutations.tolist()
        )
        self._permutation_ids = tuple(
            member_id // self.representations
            for member_id in range(ensemble_table.num_members)
        )

    def _transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        assert self._permutations is not None
        assert ensemble_table.num_groups == 1
        group = next(iter(ensemble_table))
        values = feature_coupled_cartesian(group.numerical, self._permutations)
        groups = tuple(
            group.__class__(
                columns={
                    Stype.numerical: tuple(
                        group.columns[Stype.numerical][index]
                        for index in order
                    )
                },
                numerical=values[permutation_id],
            )
            for permutation_id, order in enumerate(self._orders)
        )
        output = EnsembleTable.__new__(EnsembleTable)
        output._groups = groups
        output._locations = tuple(
            (
                permutation_id,
                ensemble_table._locations[member_id][1],
            )
            for member_id, permutation_id in enumerate(self._permutation_ids)
        )
        return output


class PlannedShiftCategories(EnsembleProcessor):
    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self, representations: int = 2) -> None:
        super().__init__()
        self.representations = representations
        self._offsets: Tensor | None = None
        self._permutation_ids: tuple[int, ...] = ()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        count = math.ceil(ensemble_table.num_members / self.representations)
        category = next(iter(ensemble_table)).categorical.categories[0]
        self._offsets = torch.randint(
            category.numel(),
            (count,),
            generator=generator,
            device=category.device,
        )
        self._permutation_ids = tuple(
            member_id // self.representations
            for member_id in range(ensemble_table.num_members)
        )

    def _transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        assert self._offsets is not None
        group = next(iter(ensemble_table))
        source = group.categorical.code[0]
        valid = source >= 0
        cardinality = group.categorical.categories[0].numel()
        mapped = (
            source.unsqueeze(0) - self._offsets[:, None, None]
        ).remainder(cardinality)
        mapped = torch.where(valid.unsqueeze(0), mapped, source.unsqueeze(0))
        category = group.categorical.categories[0]
        groups = []
        for offset, codes in zip(self._offsets, mapped, strict=True):
            order = (
                torch.arange(cardinality, device=category.device) + offset
            ).remainder(cardinality)
            groups.append(
                TableTensor(
                    categorical=CategoricalTensor(
                        code=codes.unsqueeze(0),
                        categories=(category[order],),
                    )
                )
            )
        output = EnsembleTable.__new__(EnsembleTable)
        output._groups = tuple(groups)
        output._locations = tuple(
            (permutation_id, 0) for permutation_id in self._permutation_ids
        )
        return output


def make_recipe(*, planned: bool) -> sp.Recipe:
    column_shuffle: EnsembleProcessor = (
        PlannedBatchedShuffleColumns()
        if planned
        else sp.ShuffleColumns(method="shift")
    )
    category_shuffle: EnsembleProcessor = (
        PlannedShiftCategories()
        if planned
        else sp.ShuffleCategories(method="shift")
    )
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ToNumerical(),
                ]
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.ImputeMean(),
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    column_shuffle,
                ]
            ),
        ],
        target=sp.StypeDispatch(
            categorical=[
                sp.AlignCategories(sort_by="value"),
                category_shuffle,
            ],
            numerical=sp.Standardize(),
        ),
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(classification=sp.Softmax(temperature=0.9)),
        ],
    )


def make_recipe_inputs(
    rows: int,
    columns: int,
    *,
    categorical_columns: int,
    classes: int,
    device: torch.device,
) -> tuple[TableTensor, TableTensor]:
    numerical_columns = columns - categorical_columns
    generator = torch.Generator(device=device).manual_seed(1729)
    numerical = torch.randn(
        rows, numerical_columns, generator=generator, device=device
    )
    codes = torch.randint(
        128,
        (rows, categorical_columns),
        dtype=torch.int32,
        generator=generator,
        device=device,
    )
    categories = tuple(
        torch.arange(128, device=device) for _ in range(categorical_columns)
    )
    features = TableTensor(
        numerical=numerical,
        categorical=CategoricalTensor(code=codes, categories=categories),
    )
    target_codes = torch.randint(
        classes,
        (rows, 1),
        dtype=torch.int32,
        generator=generator,
        device=device,
    )
    target_codes[: rows // 10] = -1
    target = TableTensor(
        categorical=CategoricalTensor(
            code=target_codes,
            categories=(torch.arange(classes, device=device),),
        )
    )
    return features, target


def benchmark_feature_execution(device: torch.device) -> list[dict[str, Any]]:
    results = []
    estimators = 8
    representations = 2
    source_ids = tuple(
        member_id % representations for member_id in range(estimators)
    )
    coupled_ids = tuple(
        member_id // representations for member_id in range(estimators)
    )
    independent_ids = tuple(range(estimators))

    for rows in (1_000, 10_000, 50_000):
        columns = 100
        source = torch.randn(representations, rows, columns, device=device)
        permutations = cyclic_permutations(estimators, columns, device)
        source_ids_tensor = torch.tensor(source_ids, device=device)
        independent_ids_tensor = torch.tensor(independent_ids, device=device)

        reference_coupled = feature_reference(
            source, permutations, source_ids, coupled_ids
        )
        coupled = feature_coupled_cartesian(source, permutations[:4])
        for member_id, expected in enumerate(reference_coupled):
            permutation_id = coupled_ids[member_id]
            source_id = source_ids[member_id]
            torch.testing.assert_close(
                coupled[permutation_id, source_id], expected
            )

        scenarios: tuple[tuple[str, int, Callable[[], object]], ...] = (
            (
                "per_member_reference_coupled_k4",
                4,
                lambda: feature_reference(
                    source, permutations, source_ids, coupled_ids
                ),
            ),
            (
                "batched_cartesian_coupled_k4",
                4,
                lambda: feature_coupled_cartesian(source, permutations[:4]),
            ),
            (
                "batched_by_source_independent_k8",
                8,
                lambda: feature_independent_by_source(
                    source, permutations, source_ids, independent_ids
                ),
            ),
            (
                "one_gather_source_copy_independent_k8",
                8,
                lambda: feature_independent_one_gather_with_source_copy(
                    source,
                    permutations,
                    source_ids_tensor,
                    independent_ids_tensor,
                ),
            ),
        )
        for name, unique_permutations, operation in scenarios:
            results.append(
                {
                    "suite": "feature_execution_fixed_e8_s2",
                    "scenario": name,
                    "rows": rows,
                    "columns": columns,
                    "estimators": estimators,
                    "representations": representations,
                    "unique_permutations": unique_permutations,
                    **measure(operation),
                }
            )
    return results


def benchmark_current_processor_scaling(
    device: torch.device,
) -> list[dict[str, Any]]:
    results = []
    for estimators in (1, 2, 4, 8, 16):
        ensemble = make_feature_ensemble(
            10_000,
            32,
            estimators,
            representations=1,
            device=device,
        )
        processor = sp.ShuffleColumns(method="shift").fit_ensemble(
            ensemble,
            generator=torch.Generator(device=device).manual_seed(7),
        )
        results.append(
            {
                "suite": "current_shuffle_columns_estimator_scaling",
                "scenario": "transform_ensemble",
                "rows": 10_000,
                "columns": 32,
                "estimators": estimators,
                "representations": 1,
                "unique_permutations": estimators,
                **measure(lambda: processor.transform_ensemble(ensemble)),
            }
        )
    return results


def benchmark_current_vs_planned_columns(
    device: torch.device,
) -> list[dict[str, Any]]:
    results = []
    for rows in (1_000, 10_000, 50_000):
        ensemble = make_feature_ensemble(
            rows,
            100,
            8,
            representations=2,
            device=device,
        )
        processors = {
            "current_independent_k8": sp.ShuffleColumns(
                method="shift"
            ).fit_ensemble(
                ensemble,
                generator=torch.Generator(device=device).manual_seed(7),
            ),
            "planned_batched_coupled_k4": (
                PlannedBatchedShuffleColumns().fit_ensemble(ensemble)
            ),
        }
        planned = cast(
            PlannedBatchedShuffleColumns,
            processors["planned_batched_coupled_k4"],
        )
        planned_output = planned.transform_ensemble(ensemble)
        assert planned._permutations is not None
        for member_id in range(ensemble.num_members):
            permutation_id = member_id // 2
            expected = ensemble.table(member_id).numerical.index_select(
                -1, planned._permutations[permutation_id]
            )
            torch.testing.assert_close(
                planned_output.table(member_id).numerical, expected
            )

        operations = {
            name: lambda processor=processor: processor.transform_ensemble(
                ensemble
            )
            for name, processor in processors.items()
        }
        measurements = measure_interleaved(
            operations,
            warmup_rounds=WARMUPS,
            repetitions=REPETITIONS,
        )
        for name, measurement in measurements.items():
            results.append(
                {
                    "suite": "full_shuffle_columns_processor",
                    "scenario": name,
                    "rows": rows,
                    "columns": 100,
                    "estimators": 8,
                    "representations": 2,
                    "unique_permutations": (
                        8 if name == "current_independent_k8" else 4
                    ),
                    **measurement,
                }
            )
    return results


def benchmark_target_execution(device: torch.device) -> list[dict[str, Any]]:
    results = []
    for rows in (50_000, 500_000):
        classes = 100
        generator = torch.Generator(device=device).manual_seed(1729)
        codes = torch.randint(
            classes,
            (rows, 1),
            dtype=torch.int32,
            generator=generator,
            device=device,
        )
        codes[: rows // 10] = -1
        valid = codes >= 0
        offsets = torch.arange(8, device=device, dtype=codes.dtype)

        def independent_eight() -> Tensor:
            mapped = (codes.unsqueeze(0) - offsets[:, None, None]).remainder(
                classes
            )
            return torch.where(valid.unsqueeze(0), mapped, codes.unsqueeze(0))

        def coupled_four_shared() -> Tensor:
            mapped = (codes.unsqueeze(0) - offsets[:4, None, None]).remainder(
                classes
            )
            return torch.where(valid.unsqueeze(0), mapped, codes.unsqueeze(0))

        def coupled_four_materialized_eight() -> Tensor:
            mapped = coupled_four_shared()
            member_ids = torch.arange(4, device=device).repeat_interleave(2)
            return mapped.index_select(0, member_ids)

        for name, unique, physical_outputs, operation in (
            ("independent_k8", 8, 8, independent_eight),
            ("coupled_k4_shared_storage", 4, 4, coupled_four_shared),
            (
                "coupled_k4_materialized_e8",
                4,
                8,
                coupled_four_materialized_eight,
            ),
        ):
            results.append(
                {
                    "suite": "target_shift_fixed_e8",
                    "scenario": name,
                    "rows": rows,
                    "classes": classes,
                    "estimators": 8,
                    "unique_permutations": unique,
                    "physical_outputs": physical_outputs,
                    **measure(operation),
                }
            )
    return results


def benchmark_current_vs_planned_target(
    device: torch.device,
) -> list[dict[str, Any]]:
    results = []
    for rows in (50_000, 500_000):
        classes = 100
        generator = torch.Generator(device=device).manual_seed(1729)
        codes = torch.randint(
            classes,
            (rows, 1),
            dtype=torch.int32,
            generator=generator,
            device=device,
        )
        codes[: rows // 10] = -1
        target = TableTensor(
            categorical=CategoricalTensor(
                code=codes,
                categories=(torch.arange(classes, device=device),),
            )
        )
        ensemble = EnsembleTable.__new__(EnsembleTable)
        ensemble._groups = (
            cast(
                TableTensor,
                target.unsqueeze(0).expand(8, *target.size()),
            ),
        )
        ensemble._locations = tuple((0, member_id) for member_id in range(8))
        processors = {
            "current_independent_k8": sp.ShuffleCategories(
                method="shift"
            ).fit_ensemble(
                ensemble,
                generator=torch.Generator(device=device).manual_seed(7),
            ),
            "planned_coupled_k4": PlannedShiftCategories().fit_ensemble(
                ensemble,
                generator=torch.Generator(device=device).manual_seed(7),
            ),
        }

        planned = cast(
            PlannedShiftCategories, processors["planned_coupled_k4"]
        )
        planned_output = planned.transform_ensemble(ensemble)
        for member_id in range(8):
            original_code = ensemble.table(member_id).categorical.code[:1024]
            output = planned_output.table(member_id).categorical
            output_code = output.code[:1024]
            valid = original_code >= 0
            torch.testing.assert_close(
                output.categories[0][output_code[valid].to(torch.long)],
                target.categorical.categories[0][
                    original_code[valid].to(torch.long)
                ],
            )
            assert torch.equal(output_code[~valid], original_code[~valid])

        operations = {
            name: lambda processor=processor: processor.transform_ensemble(
                ensemble
            )
            for name, processor in processors.items()
        }
        measurements = measure_interleaved(
            operations,
            warmup_rounds=WARMUPS,
            repetitions=REPETITIONS,
        )
        for name, measurement in measurements.items():
            results.append(
                {
                    "suite": "full_shuffle_categories_processor",
                    "scenario": name,
                    "rows": rows,
                    "classes": classes,
                    "estimators": 8,
                    "unique_permutations": (
                        8 if name == "current_independent_k8" else 4
                    ),
                    "physical_outputs": (
                        8 if name == "current_independent_k8" else 4
                    ),
                    **measurement,
                }
            )
    return results


def latin_reference(columns: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)

    def square(symbols: list[int]) -> list[list[int]]:
        if len(symbols) == 1:
            return [symbols]
        symbol = rng.choice(symbols)
        symbols.remove(symbol)
        rows = square(symbols)
        rows.append(rows[0].copy())
        for index, row in enumerate(rows):
            row.insert(index, symbol)
        return rows

    rows = square(list(range(columns)))
    rng.shuffle(rows)
    permutations = list(zip(*rows))
    rng.shuffle(permutations)
    return [list(permutation) for permutation in permutations]


class Fenwick:
    def __init__(self, size: int) -> None:
        self.tree = [0] + [index & -index for index in range(1, size + 1)]

    def pop(self, rank: int) -> int:
        index = 0
        bit = 1 << (len(self.tree) - 1).bit_length() - 1
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] <= rank:
                index = candidate
                rank -= self.tree[candidate]
            bit >>= 1
        position = index
        update = position + 1
        while update < len(self.tree):
            self.tree[update] -= 1
            update += update & -update
        return position


def compact_latin_plan(
    columns: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    remaining = Fenwick(columns)
    symbols = []
    for count in range(columns, 1, -1):
        symbols.append(remaining.pop(rng.randrange(count)))
    symbols.append(remaining.pop(0))
    rows = list(range(columns))
    rng.shuffle(rows)
    permutation_ids = list(range(columns))
    rng.shuffle(permutation_ids)
    return symbols, rows, permutation_ids


def materialize_compact_latin(
    plan: tuple[list[int], list[int], list[int]],
    count: int,
    device: torch.device,
) -> Tensor:
    symbols, rows, permutation_ids = plan
    columns = len(symbols)
    symbol_tensor = torch.tensor(symbols, device=device)
    row_tensor = torch.tensor(rows, device=device)
    selected = torch.tensor(permutation_ids[:count], device=device)
    return symbol_tensor[
        (selected[:, None] - row_tensor[None, :]).remainder(columns)
    ]


def benchmark_latin(device: torch.device) -> list[dict[str, Any]]:
    for seed in range(4):
        for columns in range(1, 101):
            reference = latin_reference(columns, seed)
            plan = compact_latin_plan(columns, seed)
            compact = (
                materialize_compact_latin(plan, columns, device).cpu().tolist()
            )
            assert compact == reference

    results = []
    for columns in (100, 4_000, 16_000, 64_000):
        for count in (4, 8):
            plan = compact_latin_plan(columns, 7)
            results.append(
                {
                    "suite": "exact_latin_plan",
                    "scenario": "materialize_only",
                    "columns": columns,
                    "unique_permutations": count,
                    "state_bytes": count * columns * 8,
                    **measure(
                        lambda plan=plan, count=count: (
                            materialize_compact_latin(plan, count, device)
                        )
                    ),
                }
            )
            results.append(
                {
                    "suite": "exact_latin_plan",
                    "scenario": "plan_and_materialize",
                    "columns": columns,
                    "unique_permutations": count,
                    "state_bytes": count * columns * 8,
                    **measure(
                        lambda columns=columns, count=count: (
                            materialize_compact_latin(
                                compact_latin_plan(columns, 7),
                                count,
                                device,
                            )
                        ),
                        repetitions=10,
                    ),
                }
            )

    for columns in (10, 100):
        results.append(
            {
                "suite": "exact_latin_plan",
                "scenario": "pinned_recursive_reference",
                "columns": columns,
                "unique_permutations": columns,
                **measure(
                    lambda columns=columns: latin_reference(columns, 7),
                    repetitions=10,
                ),
            }
        )
    return results


def benchmark_permutation_state_materialization(
    device: torch.device,
) -> list[dict[str, Any]]:
    results = []
    for columns in (100, 4_000, 64_000):
        base = torch.arange(columns, device=device)
        for unique_permutations in (4, 8):
            offsets = torch.arange(unique_permutations, device=device)

            def materialize() -> Tensor:
                return (base.unsqueeze(0) + offsets.unsqueeze(1)).remainder(
                    columns
                )

            results.append(
                {
                    "suite": "permutation_state_materialization",
                    "scenario": "cyclic_cuda",
                    "columns": columns,
                    "unique_permutations": unique_permutations,
                    "state_bytes": unique_permutations * columns * 8,
                    **measure(materialize),
                }
            )
    return results


def benchmark_recipe(device: torch.device) -> list[dict[str, Any]]:
    results = []
    for rows in (1_000, 3_000, 10_000, 50_000):
        features, target = make_recipe_inputs(
            rows,
            100,
            categorical_columns=10,
            classes=10,
            device=device,
        )
        executions = {
            "current_main": RecipeExecution(make_recipe(planned=False)),
            "planned_coupled_batched": RecipeExecution(
                make_recipe(planned=True)
            ),
        }
        generators = {
            name: torch.Generator(device=device).manual_seed(7)
            for name in executions
        }
        operations = {
            name: (
                lambda execution=execution, generator=generators[name]: (
                    execution.fit_transform(
                        features,
                        target,
                        None,
                        num_members=8,
                        generator=generator,
                    )
                )
            )
            for name, execution in executions.items()
        }
        for operation in operations.values():
            smoke = cast(tuple[object, ...], operation())
            assert len(smoke) == 8
        measurements = measure_interleaved(
            operations,
            warmup_rounds=WARMUPS,
            repetitions=20,
        )
        for name, measurement in measurements.items():
            results.append(
                {
                    "suite": "classification_recipe_fit_transform",
                    "scenario": name,
                    "rows": rows,
                    "columns": 100,
                    "categorical_columns": 10,
                    "classes": 10,
                    "estimators": 8,
                    **measurement,
                }
            )
    return results


def benchmark_model_end_to_end(device: torch.device) -> list[dict[str, Any]]:
    from sdm.models import TabICLv2

    context_rows = 2_400
    query_rows = 600
    columns = 100
    classes = 10
    generator = torch.Generator(device=device).manual_seed(1729)
    context = TableTensor.from_tensor(
        torch.randn(
            context_rows,
            columns,
            generator=generator,
            device=device,
        )
    )
    query = TableTensor.from_tensor(
        torch.randn(
            query_rows,
            columns,
            generator=generator,
            device=device,
        )
    )
    target = TableTensor(
        categorical=CategoricalTensor(
            code=torch.randint(
                classes,
                (context_rows, 1),
                dtype=torch.int32,
                generator=generator,
                device=device,
            ),
            categories=(torch.arange(classes, device=device),),
        )
    )
    model = TabICLv2(pretrained=False, device=device)
    model.eval()
    recipes = {
        "current_main": make_recipe(planned=False),
        "planned_coupled_batched": make_recipe(planned=True),
    }

    def operation(recipe: sp.Recipe) -> object:
        return model(
            x_context=context,
            y_context=target,
            x_query=query,
            recipe=recipe,
            num_estimators=8,
            generator=torch.Generator(device=device).manual_seed(7),
        )

    operations = {
        name: lambda recipe=recipe: operation(recipe)
        for name, recipe in recipes.items()
    }
    for run in operations.values():
        smoke = run()
        assert cast(TableTensor, smoke).size() == (query_rows, classes)

    measurements = measure_interleaved(
        operations,
        warmup_rounds=2,
        repetitions=10,
    )
    results = [
        {
            "suite": "tabiclv2_model_end_to_end",
            "scenario": name,
            "rows": context_rows + query_rows,
            "context_rows": context_rows,
            "query_rows": query_rows,
            "columns": columns,
            "classes": classes,
            "estimators": 8,
            **measurement,
        }
        for name, measurement in measurements.items()
    ]
    gc.collect()
    return results


def profile_operation(
    operation: Callable[[], object],
    *,
    warmups: int = 3,
    iterations: int = 10,
) -> dict[str, Any]:
    from collections import Counter

    from torch._C._autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    output: object | None = None
    for _ in range(warmups):
        output = operation()
    torch.cuda.synchronize()

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=False,
        profile_memory=False,
    ) as profiler:
        for _ in range(iterations):
            output = operation()
        torch.cuda.synchronize()

    events = profiler.events()
    names = Counter(event.name for event in events)
    cuda_events = sum(event.device_type == DeviceType.CUDA for event in events)
    result = {
        "profile_iterations": iterations,
        "cuda_events_per_call": cuda_events / iterations,
        "cuda_launch_calls_per_call": names["cudaLaunchKernel"] / iterations,
        "cuda_memcpy_async_calls_per_call": (
            names["cudaMemcpyAsync"] / iterations
        ),
        "selected_operator_calls_per_call": {
            name: count / iterations
            for name, count in sorted(names.items())
            if name
            in {
                "aten::cat",
                "aten::clone",
                "aten::gather",
                "aten::index_select",
                "aten::remainder",
                "aten::stack",
                "aten::where",
            }
        },
    }
    del output
    gc.collect()
    return result


def benchmark_profiles(device: torch.device) -> list[dict[str, Any]]:
    results = []
    feature_ensemble = make_feature_ensemble(
        50_000,
        100,
        8,
        representations=2,
        device=device,
    )
    current_columns = sp.ShuffleColumns(method="shift").fit_ensemble(
        feature_ensemble,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    planned_columns = PlannedBatchedShuffleColumns().fit_ensemble(
        feature_ensemble
    )
    for name, processor in (
        ("current_independent_k8", current_columns),
        ("planned_batched_coupled_k4", planned_columns),
    ):
        results.append(
            {
                "suite": "torch_profiler_shuffle_columns",
                "scenario": name,
                "rows": 50_000,
                "columns": 100,
                "estimators": 8,
                **profile_operation(
                    lambda processor=processor: processor.transform_ensemble(
                        feature_ensemble
                    )
                ),
            }
        )

    classes = 100
    codes = torch.arange(500_000, device=device, dtype=torch.int32)
    codes = codes.remainder(classes).unsqueeze(1)
    target = TableTensor(
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(classes, device=device),),
        )
    )
    target_ensemble = EnsembleTable.__new__(EnsembleTable)
    target_ensemble._groups = (
        cast(
            TableTensor,
            target.unsqueeze(0).expand(8, *target.size()),
        ),
    )
    target_ensemble._locations = tuple(
        (0, member_id) for member_id in range(8)
    )
    current_categories = sp.ShuffleCategories(method="shift").fit_ensemble(
        target_ensemble,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    planned_categories = PlannedShiftCategories().fit_ensemble(
        target_ensemble,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    for name, processor in (
        ("current_independent_k8", current_categories),
        ("planned_coupled_k4", planned_categories),
    ):
        results.append(
            {
                "suite": "torch_profiler_shuffle_categories",
                "scenario": name,
                "rows": 500_000,
                "classes": classes,
                "estimators": 8,
                **profile_operation(
                    lambda processor=processor: processor.transform_ensemble(
                        target_ensemble
                    )
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)

    results = []
    with torch.inference_mode():
        results.extend(benchmark_feature_execution(device))
        results.extend(benchmark_current_processor_scaling(device))
        results.extend(benchmark_current_vs_planned_columns(device))
        results.extend(benchmark_target_execution(device))
        results.extend(benchmark_current_vs_planned_target(device))
        results.extend(benchmark_latin(device))
        results.extend(benchmark_permutation_state_materialization(device))
        results.extend(benchmark_recipe(device))
        results.extend(benchmark_model_end_to_end(device))
        results.extend(benchmark_profiles(device))

    payload = {
        "metadata": {
            "generated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "baseline_commit": BASELINE_COMMIT,
            "reference_commit": REFERENCE_COMMIT,
            "candidate": "benchmark-only planned coupled batched processors",
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
        },
        "methodology": {
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "timing": "synchronized wall clock and CUDA events",
            "memory": "maximum allocated bytes above pre-operation baseline",
            "input_generation": "excluded",
            "fitting": "excluded from transform suites; included in RecipeExecution.fit_transform",
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"metadata": payload["metadata"], "results": len(results)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
