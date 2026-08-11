# ruff: noqa: E501, T201

"""Adaptively characterize the runtime drivers of every public Processor.

The benchmark varies one input characteristic at a time.  It first measures
three pilot points and only spends the full benchmark budget when the observed
effect exceeds the device-specific noise floor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from sdm import CategoricalTensor, NaT, StringTensor, Stype, TableTensor
from sdm.processing import (
    PCA,
    TFIDF,
    AddCalendarFields,
    AlignCategories,
    Choice,
    Clip,
    ClipQuantiles,
    ClipSigma,
    DropConstantColumns,
    EmbedText,
    Identity,
    ImputeMean,
    ImputeMode,
    PowerTransform,
    QuantileTransform,
    ReduceEstimators,
    SelectColumns,
    Sequential,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
)
from sdm.processing import (
    Callable as CallableProcessor,
)
from sdm.tensor import EnsembleTable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmark/results/processor_characterization.json"
DEFAULT_PLOTS = ROOT / "benchmark/plots"


@dataclass(frozen=True)
class Axis:
    """One controlled input characteristic and its candidate values."""

    name: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class Sweep:
    """One Processor operation characterized along one relevant axis."""

    processor: str
    operation: str
    family: str
    stypes: tuple[str, ...]
    axis: Axis


@dataclass
class Measurement:
    """Repeated timing and CUDA allocation measurements for one point."""

    median_ms: float
    p95_ms: float
    minimum_ms: float
    cuda_event_median_ms: float | None
    cuda_enqueue_median_ms: float | None
    cuda_sync_wait_median_ms: float | None
    peak_cuda_bytes: int | None
    repetitions: int


AXES = {
    "rows": Axis("rows", (1_000, 4_000, 10_000, 25_000, 50_000)),
    "columns": Axis("columns", (4, 16, 32, 64, 128)),
    "dtype": Axis("dtype", ("float32", "float64")),
    "outlier_ratio": Axis("outlier_ratio", (0.0, 0.001, 0.01, 0.1, 0.5)),
    "constant_ratio": Axis("constant_ratio", (0.0, 0.1, 0.25, 0.5, 1.0)),
    "missing_ratio": Axis("missing_ratio", (0.0, 0.01, 0.1, 0.5, 0.9)),
    "categorical_columns": Axis("categorical_columns", (1, 4, 8, 16, 32)),
    "numerical_columns": Axis("numerical_columns", (0, 4, 8, 16, 32)),
    "cardinality": Axis("cardinality", (16, 128, 1_024, 4_096, 16_384)),
    "unseen_ratio": Axis("unseen_ratio", (0.0, 0.01, 0.1, 0.5, 0.9)),
    "code_dtype": Axis("code_dtype", ("int32", "int64")),
    "category_type": Axis("category_type", ("integer", "string")),
    "estimators": Axis("estimators", (1, 2, 4, 8, 16)),
    "output_width": Axis("output_width", (2, 8, 32, 128, 512)),
    "components": Axis("components", (2, 8, 16, 32, 64)),
    "steps": Axis("steps", (1, 2, 4, 8, 16)),
    "options": Axis("options", (1, 2, 4, 8, 16)),
    "routes": Axis("routes", (1, 2, 3)),
    "selected_fraction": Axis(
        "selected_fraction", (0.1, 0.25, 0.5, 0.75, 1.0)
    ),
    "text_columns": Axis("text_columns", (1, 2, 4, 8)),
    "text_length": Axis("text_length", (8, 16, 32, 64, 128)),
    "vocabulary": Axis("vocabulary", (8, 32, 128, 512, 2_048)),
    "embedding_dim": Axis("embedding_dim", (8, 32, 64, 128, 512)),
    "field_count": Axis("field_count", (1, 2, 4, 5)),
}


DEFAULTS: dict[str, Any] = {
    "rows": 10_000,
    "columns": 32,
    "dtype": "float32",
    "outlier_ratio": 0.01,
    "constant_ratio": 0.1,
    "missing_ratio": 0.01,
    "categorical_columns": 8,
    "numerical_columns": 8,
    "cardinality": 1_024,
    "unseen_ratio": 0.01,
    "code_dtype": "int32",
    "category_type": "integer",
    "estimators": 8,
    "output_width": 32,
    "components": 16,
    "steps": 4,
    "options": 4,
    "routes": 3,
    "selected_fraction": 0.5,
    "text_columns": 1,
    "text_length": 32,
    "vocabulary": 128,
    "embedding_dim": 64,
    "field_count": 4,
}


PROCESSOR_META: dict[str, dict[str, str]] = {
    "Identity": {
        "frequency": "high",
        "bottleneck": "orchestration",
        "priority": "none",
    },
    "Callable": {
        "frequency": "medium",
        "bottleneck": "callable-dependent",
        "priority": "none",
    },
    "Sequential": {
        "frequency": "high",
        "bottleneck": "orchestration",
        "priority": "low",
    },
    "StypeDispatch": {
        "frequency": "high",
        "bottleneck": "orchestration",
        "priority": "low",
    },
    "TaskDispatch": {
        "frequency": "high",
        "bottleneck": "orchestration",
        "priority": "low",
    },
    "Choice": {
        "frequency": "high",
        "bottleneck": "allocations and orchestration",
        "priority": "medium",
    },
    "ToNumerical": {
        "frequency": "high",
        "bottleneck": "data movement",
        "priority": "medium",
    },
    "ShuffleColumns": {
        "frequency": "high",
        "bottleneck": "allocations and data movement",
        "priority": "medium",
    },
    "SelectColumns": {
        "frequency": "medium",
        "bottleneck": "metadata and views",
        "priority": "low",
    },
    "TFIDF": {
        "frequency": "low",
        "bottleneck": "computation and allocations",
        "priority": "high",
    },
    "EmbedText": {
        "frequency": "low",
        "bottleneck": "external model and data conversion",
        "priority": "workload-dependent",
    },
    "Clip": {
        "frequency": "medium",
        "bottleneck": "memory bandwidth",
        "priority": "low",
    },
    "ClipQuantiles": {
        "frequency": "medium",
        "bottleneck": "computation",
        "priority": "medium",
    },
    "ClipSigma": {
        "frequency": "medium",
        "bottleneck": "memory bandwidth",
        "priority": "medium",
    },
    "ImputeMean": {
        "frequency": "high",
        "bottleneck": "memory bandwidth",
        "priority": "medium",
    },
    "PowerTransform": {
        "frequency": "high",
        "bottleneck": "CPU computation; GPU launch/orchestration overhead",
        "priority": "high",
    },
    "QuantileTransform": {
        "frequency": "medium",
        "bottleneck": "sorting/search and per-chunk orchestration",
        "priority": "high",
    },
    "Standardize": {
        "frequency": "high",
        "bottleneck": "memory bandwidth",
        "priority": "medium",
    },
    "DropConstantColumns": {
        "frequency": "high",
        "bottleneck": "synchronization and metadata",
        "priority": "medium",
    },
    "PCA": {
        "frequency": "low",
        "bottleneck": "computation",
        "priority": "workload-dependent",
    },
    "AlignCategories": {
        "frequency": "high",
        "bottleneck": "per-column orchestration and allocations",
        "priority": "high",
    },
    "ShuffleCategories": {
        "frequency": "high",
        "bottleneck": "per-column/per-estimator orchestration and allocations",
        "priority": "high",
    },
    "ImputeMode": {
        "frequency": "high",
        "bottleneck": "allocations and memory bandwidth",
        "priority": "medium",
    },
    "AddCalendarFields": {
        "frequency": "medium",
        "bottleneck": "repeated integer computation and allocations",
        "priority": "high on CPU",
    },
    "ReduceEstimators": {
        "frequency": "high",
        "bottleneck": "memory bandwidth",
        "priority": "high",
    },
    "Softmax": {
        "frequency": "high",
        "bottleneck": "computation and memory bandwidth",
        "priority": "medium",
    },
}


def _make_sweeps() -> list[Sweep]:
    sweeps: list[Sweep] = []

    def add(
        processor: str,
        operation: str,
        family: str,
        stypes: Sequence[str],
        axes: Sequence[str],
    ) -> None:
        sweeps.extend(
            Sweep(processor, operation, family, tuple(stypes), AXES[axis])
            for axis in axes
        )

    add("Identity", "transform", "numerical", ("numerical",), ("rows",))
    add("Callable", "transform", "numerical", ("any",), ("rows",))
    add(
        "Sequential",
        "transform_ensemble",
        "numerical",
        ("any",),
        ("steps", "estimators"),
    )
    add(
        "StypeDispatch",
        "transform_ensemble",
        "mixed",
        ("all",),
        ("routes", "estimators"),
    )
    add(
        "TaskDispatch",
        "transform_ensemble",
        "numerical",
        ("any",),
        ("estimators",),
    )
    add(
        "Choice",
        "fit_transform_ensemble",
        "numerical",
        ("any",),
        ("rows", "options", "estimators"),
    )
    add(
        "ToNumerical",
        "transform",
        "mixed",
        ("categorical", "numerical"),
        ("rows", "categorical_columns", "numerical_columns", "dtype"),
    )
    add(
        "ShuffleColumns",
        "fit_transform_ensemble",
        "numerical",
        ("any",),
        ("rows", "columns", "estimators", "dtype"),
    )
    add(
        "SelectColumns",
        "transform",
        "mixed",
        ("all",),
        ("columns", "selected_fraction"),
    )

    add(
        "TFIDF",
        "fit_transform",
        "text",
        ("text",),
        ("rows", "text_columns", "text_length", "vocabulary", "missing_ratio"),
    )
    add(
        "TFIDF",
        "fit_transform_ensemble",
        "text",
        ("text",),
        ("estimators",),
    )
    add(
        "EmbedText",
        "transform",
        "text",
        ("text",),
        ("rows", "text_columns", "text_length", "embedding_dim"),
    )

    add(
        "Clip",
        "transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype"),
    )
    add(
        "ClipQuantiles",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "constant_ratio"),
    )
    add(
        "ClipSigma",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "outlier_ratio"),
    )
    add(
        "ImputeMean",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "missing_ratio"),
    )
    add(
        "PowerTransform",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "outlier_ratio", "constant_ratio"),
    )
    add(
        "QuantileTransform",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "constant_ratio"),
    )
    add(
        "Standardize",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "constant_ratio"),
    )
    add(
        "DropConstantColumns",
        "fit_transform_ensemble",
        "numerical",
        ("numerical",),
        ("rows", "columns", "constant_ratio", "estimators"),
    )
    add(
        "PCA",
        "fit_transform",
        "numerical",
        ("numerical",),
        ("rows", "columns", "dtype", "components"),
    )

    add(
        "AlignCategories",
        "fit_transform",
        "categorical",
        ("categorical",),
        (
            "rows",
            "categorical_columns",
            "cardinality",
            "missing_ratio",
            "code_dtype",
            "category_type",
        ),
    )
    add(
        "AlignCategories",
        "fit_transform_ensemble",
        "categorical",
        ("categorical",),
        ("estimators",),
    )
    add(
        "AlignCategories",
        "transform",
        "categorical",
        ("categorical",),
        ("unseen_ratio",),
    )
    add(
        "ShuffleCategories",
        "fit_transform_ensemble",
        "categorical",
        ("categorical",),
        (
            "rows",
            "categorical_columns",
            "cardinality",
            "missing_ratio",
            "estimators",
        ),
    )
    add(
        "ImputeMode",
        "fit_transform",
        "categorical",
        ("categorical",),
        (
            "rows",
            "categorical_columns",
            "cardinality",
            "missing_ratio",
            "code_dtype",
        ),
    )

    add(
        "AddCalendarFields",
        "transform",
        "datetime",
        ("datetime",),
        ("rows", "columns", "field_count", "missing_ratio"),
    )
    add(
        "ReduceEstimators",
        "transform",
        "output",
        ("numerical",),
        ("rows", "output_width", "estimators", "dtype"),
    )
    add(
        "Softmax",
        "transform",
        "output",
        ("numerical",),
        ("rows", "output_width", "estimators", "dtype"),
    )
    return sweeps


SWEEPS = _make_sweeps()


def _seed(*parts: Any) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFF_FFFF


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _columns(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(count))


def _apply_ratio(flat: torch.Tensor, ratio: float, value: float | int) -> None:
    count = round(flat.numel() * ratio)
    if count:
        flat[:count] = value


def _numerical_table(
    params: dict[str, Any], device: torch.device
) -> TableTensor:
    rows = int(params["rows"])
    columns = int(params["columns"])
    dtype = _torch_dtype(str(params["dtype"]))
    generator = torch.Generator().manual_seed(
        _seed("numerical", rows, columns, dtype)
    )
    data = torch.randn((rows, columns), generator=generator, dtype=dtype)
    constant_columns = round(columns * float(params["constant_ratio"]))
    if constant_columns:
        data[:, :constant_columns] = 1.0
    _apply_ratio(data.flatten(), float(params["outlier_ratio"]), 1_000.0)
    _apply_ratio(data.flatten(), float(params["missing_ratio"]), math.nan)
    data = data.to(device)
    return TableTensor.from_tensor(data, _columns("num", columns))


def _categorical_values(
    params: dict[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    rows = int(params["rows"])
    columns = int(params["categorical_columns"])
    cardinality = int(params["cardinality"])
    dtype = _torch_dtype(str(params["code_dtype"]))
    generator = torch.Generator().manual_seed(
        _seed("categorical", rows, columns, cardinality)
    )
    data = torch.randint(
        0, cardinality, (rows, columns), generator=generator, dtype=dtype
    )
    _apply_ratio(data.flatten(), float(params["missing_ratio"]), -1)
    categories = tuple(
        torch.arange(cardinality, dtype=torch.int64) for _ in range(columns)
    )
    return data, categories


def _string_categories(
    params: dict[str, Any], device: torch.device
) -> TableTensor:
    columns = int(params["categorical_columns"])
    cardinality = int(params["cardinality"])
    data, _ = _categorical_values(params)
    categories = tuple(
        StringTensor.from_list(
            [f"category_{index}" for index in range(cardinality)],
            device=device,
        )
        for _ in range(columns)
    )
    return TableTensor(
        columns={Stype.categorical: _columns("cat", columns)},
        categorical=CategoricalTensor(
            code=data.to(device),
            categories=categories,
        ),
    )


def _categorical_table(
    params: dict[str, Any],
    device: torch.device,
    *,
    query: bool = False,
) -> TableTensor:
    if params["category_type"] == "string":
        return _string_categories(params, device)
    data, categories = _categorical_values(params)
    if query and float(params["unseen_ratio"]):
        cardinality = int(params["cardinality"])
        _apply_ratio(
            data.flatten(), float(params["unseen_ratio"]), cardinality
        )
        categories = tuple(
            torch.arange(cardinality + 1, dtype=torch.int64)
            for _ in categories
        )
    data = data.to(device)
    categories = tuple(category.to(device) for category in categories)
    return TableTensor(
        columns={Stype.categorical: _columns("cat", data.size(-1))},
        categorical=CategoricalTensor(code=data, categories=categories),
    )


def _mixed_table(params: dict[str, Any], device: torch.device) -> TableTensor:
    rows = int(params["rows"])
    numerical_columns = int(params["numerical_columns"])
    categorical_columns = int(params["categorical_columns"])
    routes = int(params["routes"])
    dtype = _torch_dtype(str(params["dtype"]))
    numerical: torch.Tensor | None = None
    categorical: CategoricalTensor | None = None
    datetime_values: torch.Tensor | None = None
    columns: dict[Stype, tuple[str, ...]] = {}
    if numerical_columns:
        numerical = torch.arange(
            rows * numerical_columns, dtype=dtype, device=device
        ).reshape(rows, numerical_columns)
        columns[Stype.numerical] = _columns("num", numerical_columns)
    if categorical_columns and routes >= 2:
        cardinality = int(params["cardinality"])
        codes = torch.arange(
            rows * categorical_columns, device=device
        ).reshape(rows, categorical_columns)
        codes = (codes % cardinality).to(
            _torch_dtype(str(params["code_dtype"]))
        )
        categorical = CategoricalTensor(
            code=codes,
            categories=tuple(
                torch.arange(cardinality, device=device)
                for _ in range(categorical_columns)
            ),
        )
        columns[Stype.categorical] = _columns("cat", categorical_columns)
    if routes >= 3:
        datetime_columns = max(1, numerical_columns // 4)
        datetime_values = (
            1_577_836_800_000_000
            + torch.arange(
                rows * datetime_columns, dtype=torch.int64, device=device
            ).reshape(rows, datetime_columns)
            * 3_600_000_000
        )
        columns[Stype.datetime] = _columns("date", datetime_columns)
    return TableTensor(
        columns=columns,
        numerical=numerical,
        categorical=categorical,
        datetime=datetime_values,
    )


def _datetime_table(
    params: dict[str, Any], device: torch.device
) -> TableTensor:
    rows = int(params["rows"])
    columns = int(params["columns"])
    values = (
        1_577_836_800_000_000
        + torch.arange(rows * columns, dtype=torch.int64).reshape(
            rows, columns
        )
        * 3_600_000_000
    )
    _apply_ratio(values.flatten(), float(params["missing_ratio"]), NaT)
    return TableTensor(
        columns={Stype.datetime: _columns("date", columns)},
        datetime=values.to(device),
    )


def _text_table(params: dict[str, Any], device: torch.device) -> TableTensor:
    rows = int(params["rows"])
    columns = int(params["text_columns"])
    length = int(params["text_length"])
    vocabulary = int(params["vocabulary"])
    missing = float(params["missing_ratio"])
    values: list[list[str | None]] = []
    for row in range(rows):
        record: list[str | None] = []
        for column in range(columns):
            offset = row * columns + column
            if offset < rows * columns * missing:
                record.append(None)
            else:
                token = f"token_{offset % vocabulary}_"
                record.append((token * (length // len(token) + 1))[:length])
        values.append(record)
    return TableTensor.from_tensor(
        StringTensor.from_list(values, device=device),
        _columns("text", columns),
    )


def _output_table(params: dict[str, Any], device: torch.device) -> TableTensor:
    estimators = int(params["estimators"])
    rows = int(params["rows"])
    width = int(params["output_width"])
    dtype = _torch_dtype(str(params["dtype"]))
    generator = torch.Generator().manual_seed(
        _seed("output", estimators, rows, width, dtype)
    )
    data = torch.randn(
        (estimators, rows, width), generator=generator, dtype=dtype
    ).to(device)
    return TableTensor.from_tensor(data, _columns("out", width))


class _DeterministicEmbedding(torch.nn.Module):
    def __init__(self, dimensions: int, device: torch.device) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.device = device

    def forward(self, values: Any) -> torch.Tensor:
        return torch.zeros(
            (len(values), self.dimensions),
            dtype=torch.float32,
            device=self.device,
        )


def _processor(name: str, params: dict[str, Any], device: torch.device) -> Any:
    if name == "Identity":
        return Identity()
    if name == "Callable":
        return CallableProcessor(lambda table: table)
    if name == "Sequential":
        return Sequential(*(Identity() for _ in range(int(params["steps"]))))
    if name == "StypeDispatch":
        routes = int(params["routes"])
        return StypeDispatch(
            numerical=Identity(),
            categorical=Identity() if routes >= 2 else None,
            datetime=Identity() if routes >= 3 else None,
        )
    if name == "TaskDispatch":
        processor = TaskDispatch(
            classification=Identity(), regression=Identity()
        )
        target = TableTensor.from_tensor(
            torch.zeros((2, 1), dtype=torch.int32, device=device),
            columns=("target",),
        )
        processor._resolve(target)
        return processor
    if name == "Choice":
        return Choice(
            *(Identity() for _ in range(int(params["options"]))),
            selection="round_robin",
        )
    if name == "ToNumerical":
        return ToNumerical()
    if name == "ShuffleColumns":
        return ShuffleColumns(method="random")
    if name == "SelectColumns":
        count = max(
            1,
            round(int(params["columns"]) * float(params["selected_fraction"])),
        )
        return SelectColumns(max_columns=count)
    if name == "TFIDF":
        return TFIDF(ngram_range=(2, 4), max_features=1_000)
    if name == "EmbedText":
        dimensions = int(params["embedding_dim"])
        return EmbedText(
            _DeterministicEmbedding(dimensions, device), dimensions
        )
    if name == "Clip":
        return Clip(min_value=-3.0, max_value=3.0)
    if name == "ClipQuantiles":
        return ClipQuantiles(q_low=0.01, q_high=0.99)
    if name == "ClipSigma":
        return ClipSigma(threshold=3.0)
    if name == "ImputeMean":
        return ImputeMean()
    if name == "PowerTransform":
        return PowerTransform()
    if name == "QuantileTransform":
        return QuantileTransform(n_quantiles=1_000, subsample=10_000)
    if name == "Standardize":
        return Standardize()
    if name == "DropConstantColumns":
        return DropConstantColumns()
    if name == "PCA":
        return PCA(
            num_components=min(
                int(params["components"]), int(params["columns"])
            )
        )
    if name == "AlignCategories":
        return AlignCategories()
    if name == "ShuffleCategories":
        return ShuffleCategories(method="random")
    if name == "ImputeMode":
        return ImputeMode()
    if name == "AddCalendarFields":
        fields = ("minute", "hour", "weekday", "day_of_month", "month")
        return AddCalendarFields(fields[: int(params["field_count"])])
    if name == "ReduceEstimators":
        return ReduceEstimators()
    if name == "Softmax":
        return Softmax()
    raise KeyError(name)


def _table(
    sweep: Sweep,
    params: dict[str, Any],
    device: torch.device,
    *,
    query: bool = False,
) -> TableTensor:
    if sweep.family == "numerical":
        return _numerical_table(params, device)
    if sweep.family == "categorical":
        return _categorical_table(params, device, query=query)
    if sweep.family == "mixed":
        mixed_params = dict(params)
        if sweep.processor == "ToNumerical":
            mixed_params["routes"] = 2
        elif sweep.processor == "SelectColumns":
            mixed_params["numerical_columns"] = params["columns"]
            mixed_params["categorical_columns"] = params["columns"]
            mixed_params["routes"] = 3
        return _mixed_table(mixed_params, device)
    if sweep.family == "datetime":
        return _datetime_table(params, device)
    if sweep.family == "text":
        return _text_table(params, device)
    if sweep.family == "output":
        return _output_table(params, device)
    raise KeyError(sweep.family)


def _ensemble(table: TableTensor, params: dict[str, Any]) -> EnsembleTable:
    return EnsembleTable(table, num_members=int(params["estimators"]))


def _prepare(
    sweep: Sweep,
    params: dict[str, Any],
    device: torch.device,
) -> Callable[[], Callable[[], Any]]:
    table = _table(sweep, params, device)
    operation = sweep.operation

    if operation == "transform":
        if sweep.processor == "AlignCategories":
            fitted = _processor(sweep.processor, params, device)
            fitted.fit(table)
            query = _table(sweep, params, device, query=True)
            return lambda: lambda: fitted.transform(query)
        processor = _processor(sweep.processor, params, device)
        return lambda: lambda: processor.transform(table)

    if operation == "fit_transform":

        def prepare_fit_transform() -> Callable[[], Any]:
            processor = _processor(sweep.processor, params, device)
            return lambda: processor.fit_transform(table)

        return prepare_fit_transform

    ensemble = _ensemble(table, params)
    if operation == "transform_ensemble":
        processor = _processor(sweep.processor, params, device)
        return lambda: lambda: processor.transform_ensemble(ensemble)
    if operation == "fit_transform_ensemble":

        def prepare_fit_transform_ensemble() -> Callable[[], Any]:
            processor = _processor(sweep.processor, params, device)
            return lambda: processor.fit_transform_ensemble(ensemble)

        return prepare_fit_transform_ensemble
    raise KeyError(operation)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _measure(
    prepare: Callable[[], Callable[[], Any]],
    device: torch.device,
    *,
    warmups: int,
    repetitions: int,
) -> Measurement:
    for _ in range(warmups):
        run = prepare()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    wall: list[float] = []
    cuda_events: list[float] = []
    enqueue: list[float] = []
    sync_wait: list[float] = []
    peak_cuda = 0
    for _ in range(repetitions):
        run = prepare()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        result = run()
        enqueued = time.perf_counter()
        if device.type == "cuda":
            end_event.record()
            sync_start = time.perf_counter()
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
            cuda_events.append(start_event.elapsed_time(end_event))
            enqueue.append((enqueued - start) * 1_000)
            sync_wait.append((finished - sync_start) * 1_000)
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_allocated(device))
        else:
            finished = enqueued
        wall.append((finished - start) * 1_000)
        del result

    return Measurement(
        median_ms=statistics.median(wall),
        p95_ms=_percentile(wall, 0.95),
        minimum_ms=min(wall),
        cuda_event_median_ms=statistics.median(cuda_events)
        if cuda_events
        else None,
        cuda_enqueue_median_ms=statistics.median(enqueue) if enqueue else None,
        cuda_sync_wait_median_ms=statistics.median(sync_wait)
        if sync_wait
        else None,
        peak_cuda_bytes=peak_cuda or None,
        repetitions=repetitions,
    )


def _pilot_indices(count: int) -> list[int]:
    if count <= 3:
        return list(range(count))
    return sorted({0, count // 2, count - 1})


def _effect(
    measurements: list[dict[str, Any]], device: str
) -> tuple[str, float, float, float]:
    runtimes = [point["measurement"]["median_ms"] for point in measurements]
    low = min(runtimes)
    high = max(runtimes)
    delta = high - low
    ratio = high / max(low, 1e-9)
    absolute_floor = 0.03 if device == "cuda" else 0.20
    negligible_ceiling = max(absolute_floor, low * 0.15)
    if delta <= negligible_ceiling:
        impact = "negligible"
    elif ratio >= 2.0 or (
        ratio >= 1.25 and delta >= (1.0 if device == "cuda" else 10.0)
    ):
        impact = "strong"
    else:
        impact = "moderate"
    return impact, ratio, delta, negligible_ceiling


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scaling(points: list[dict[str, Any]], impact: str) -> str:
    if impact == "negligible":
        return "flat"
    pairs = [
        (
            _numeric_value(point["value"]),
            float(point["measurement"]["median_ms"]),
        )
        for point in points
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and x > 0 and y > 0]
    if len(pairs) < 3:
        return "categorical"
    pairs.sort()
    slopes = [
        math.log(y2 / y1) / math.log(x2 / x1)
        for (x1, y1), (x2, y2) in pairwise(pairs)
        if x2 != x1 and y2 != y1
    ]
    if not slopes:
        return "flat"
    adjacent_ratios = [y2 / y1 for (_, y1), (_, y2) in pairwise(pairs)]
    if max(slopes) - min(slopes) > 1.0 and max(adjacent_ratios) > 1.5:
        return "threshold"
    slope = statistics.median(slopes)
    if slope < 0.75:
        return "sub-linear"
    if slope <= 1.25:
        return "linear"
    return "super-linear"


def _supported(sweep: Sweep, device: torch.device) -> tuple[bool, str | None]:
    if device.type == "cuda" and sweep.family == "text":
        return False, "GPU text processing requires the optional cuDF stack"
    if device.type == "cuda" and sweep.axis.name == "category_type":
        return False, "GPU string categories require the optional cuDF stack"
    return True, None


def _run_sweep(
    sweep: Sweep,
    device: torch.device,
    *,
    pilot_repetitions: int,
    full_repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    supported, reason = _supported(sweep, device)
    result: dict[str, Any] = {
        "processor": sweep.processor,
        "operation": sweep.operation,
        "stypes": list(sweep.stypes),
        "axis": sweep.axis.name,
        "device": device.type,
        "values": list(sweep.axis.values),
        "points": [],
    }
    if not supported:
        result.update({"status": "skipped", "reason": reason})
        return result

    pilot_indices = _pilot_indices(len(sweep.axis.values))
    for index in pilot_indices:
        params = dict(DEFAULTS)
        if sweep.family == "numerical" and sweep.processor != "ImputeMean":
            params["missing_ratio"] = 0.0
        params[sweep.axis.name] = sweep.axis.values[index]
        prepare = _prepare(sweep, params, device)
        measurement = _measure(
            prepare,
            device,
            warmups=warmups,
            repetitions=pilot_repetitions,
        )
        result["points"].append(
            {
                "index": index,
                "value": sweep.axis.values[index],
                "phase": "pilot",
                "measurement": asdict(measurement),
            }
        )

    impact, ratio, delta, ceiling = _effect(result["points"], device.type)
    early_stopped = impact == "negligible" and len(sweep.axis.values) > len(
        pilot_indices
    )
    if not early_stopped:
        for index, value in enumerate(sweep.axis.values):
            if index in pilot_indices:
                continue
            params = dict(DEFAULTS)
            if sweep.family == "numerical" and sweep.processor != "ImputeMean":
                params["missing_ratio"] = 0.0
            params[sweep.axis.name] = value
            prepare = _prepare(sweep, params, device)
            measurement = _measure(
                prepare,
                device,
                warmups=warmups,
                repetitions=full_repetitions,
            )
            result["points"].append(
                {
                    "index": index,
                    "value": value,
                    "phase": "expanded",
                    "measurement": asdict(measurement),
                }
            )
        result["points"].sort(key=lambda point: point["index"])
        impact, ratio, delta, ceiling = _effect(result["points"], device.type)

    result.update(
        {
            "status": "completed",
            "impact": impact,
            "runtime_ratio": ratio,
            "runtime_delta_ms": delta,
            "negligible_ceiling_ms": ceiling,
            "scaling": _scaling(result["points"], impact),
            "early_stopped": early_stopped,
            "measured_point_count": len(result["points"]),
        }
    )
    return result


def _hardware(devices: list[torch.device]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    if any(device.type == "cuda" for device in devices):
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "cuda": torch.version.cuda,
        }
    return payload


def _impact_rank(impact: str) -> int:
    return {"negligible": 0, "moderate": 1, "strong": 2}.get(impact, -1)


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _write_heatmap(payload: dict[str, Any], destination: Path) -> None:
    device = "cuda"
    completed = [
        result
        for result in payload["sweeps"]
        if result["status"] == "completed" and result["device"] == device
    ]
    axes = sorted({result["axis"] for result in completed})
    processors = sorted(PROCESSOR_META)
    cell_width = 82
    cell_height = 25
    left = 175
    top = 230
    width = left + cell_width * len(axes) + 20
    height = top + cell_height * len(processors) + 50
    impact: dict[tuple[str, str], str] = {}
    for result in completed:
        key = (result["processor"], result["axis"])
        current = impact.get(key, "")
        if _impact_rank(result["impact"]) > _impact_rank(current):
            impact[key] = result["impact"]
    colors = {
        "strong": "#d9534f",
        "moderate": "#f0c84b",
        "negligible": "#f2f2f2",
        "": "#ffffff",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.label{font-size:11px}.title{font-size:18px;font-weight:bold}.legend{font-size:12px}</style>",
        '<text x="12" y="24" class="title">GPU Processor runtime sensitivity by input characteristic</text>',
        '<text x="12" y="48" class="legend">Red (strong): slowest median is ≥2x fastest, or ≥1.25x with Δ ≥1 ms.</text>',
        '<text x="12" y="68" class="legend">Grey (negligible): Δ ≤ max(15% of fastest median, 0.03 ms).</text>',
        '<text x="12" y="88" class="legend">Yellow (moderate): between strong and negligible. White: irrelevant or not measured on GPU.</text>',
        '<text x="12" y="108" class="legend">Color measures runtime sensitivity, not optimization priority.</text>',
    ]
    for column, axis in enumerate(axes):
        x = left + column * cell_width + cell_width / 2
        lines.append(
            f'<text x="{x}" y="{top - 8}" class="label" text-anchor="start" transform="rotate(-55 {x} {top - 8})">{_svg_escape(axis)}</text>'
        )
    for row, processor in enumerate(processors):
        y = top + row * cell_height
        lines.append(
            f'<text x="8" y="{y + 17}" class="label">{_svg_escape(processor)}</text>'
        )
        for column, axis in enumerate(axes):
            x = left + column * cell_width
            value = impact.get((processor, axis), "")
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 2}" height="{cell_height - 2}" fill="{colors[value]}" stroke="#ddd"><title>{processor}: {axis}: {value or "irrelevant/not measured"} on GPU</title></rect>'
            )
    lines.append("</svg>")
    destination.write_text("\n".join(lines) + "\n")


def _write_scaling(payload: dict[str, Any], destination: Path) -> None:
    candidates = []
    for device in ("cpu", "cuda"):
        device_candidates = [
            result
            for result in payload["sweeps"]
            if result["status"] == "completed"
            and result["device"] == device
            and result["impact"] in {"strong", "moderate"}
            and len(result["points"]) >= 3
            and all(
                _numeric_value(point["value"]) is not None
                for point in result["points"]
            )
        ]
        device_candidates.sort(
            key=lambda result: result["runtime_delta_ms"], reverse=True
        )
        candidates.extend(device_candidates[:6])
    panel_width = 340
    panel_height = 220
    columns = 3
    rows = math.ceil(len(candidates) / columns) or 1
    width = columns * panel_width
    height = 45 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:18px;font-weight:bold}.panel{font-size:12px;font-weight:bold}.tick{font-size:10px}</style>",
        '<text x="12" y="25" class="title">Largest measured scaling effects</text>',
    ]
    for index, result in enumerate(candidates):
        panel_x = (index % columns) * panel_width
        panel_y = 45 + (index // columns) * panel_height
        plot_x = panel_x + 48
        plot_y = panel_y + 28
        plot_w = panel_width - 72
        plot_h = panel_height - 62
        points = sorted(
            result["points"], key=lambda point: float(point["value"])
        )
        xs = [float(point["value"]) for point in points]
        ys = [float(point["measurement"]["median_ms"]) for point in points]
        x_low, x_high = min(xs), max(xs)
        y_low, y_high = min(ys), max(ys)
        if y_high == y_low:
            y_high = y_low + 1.0
        coords = []
        for x, y in zip(xs, ys):
            px = plot_x + (x - x_low) / max(x_high - x_low, 1e-9) * plot_w
            py = plot_y + plot_h - (y - y_low) / (y_high - y_low) * plot_h
            coords.append((px, py))
        title = (
            f"{result['processor']} · {result['axis']} · {result['device']}"
        )
        lines.extend(
            [
                f'<text x="{panel_x + 8}" y="{panel_y + 14}" class="panel">{_svg_escape(title)}</text>',
                f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#888"/>',
                f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#888"/>',
                f'<text x="{plot_x - 5}" y="{plot_y + 4}" class="tick" text-anchor="end">{y_high:.2f} ms</text>',
                f'<text x="{plot_x - 5}" y="{plot_y + plot_h}" class="tick" text-anchor="end">{y_low:.2f}</text>',
                f'<text x="{plot_x}" y="{plot_y + plot_h + 15}" class="tick">{_svg_escape(points[0]["value"])}</text>',
                f'<text x="{plot_x + plot_w}" y="{plot_y + plot_h + 15}" class="tick" text-anchor="end">{_svg_escape(points[-1]["value"])}</text>',
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coords)}" fill="none" stroke="#2f6f9f" stroke-width="2"/>',
            ]
        )
        lines.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2f6f9f"/>'
            for x, y in coords
        )
    lines.append("</svg>")
    destination.write_text("\n".join(lines))


def _write_plots(payload: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_heatmap(payload, directory / "processor_characteristic_heatmap.svg")
    _write_scaling(payload, directory / "processor_scaling.svg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument(
        "--processors", nargs="+", help="Optional public Processor names"
    )
    parser.add_argument("--pilot-repetitions", type=int, default=5)
    parser.add_argument("--full-repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected adaptive sweeps and write raw data plus plots."""
    args = _parse_args()
    requested_devices: list[torch.device] = []
    for name in args.devices:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        requested_devices.append(torch.device(name))
    selected = set(args.processors or PROCESSOR_META)
    unknown = selected.difference(PROCESSOR_META)
    if unknown:
        raise ValueError(f"Unknown Processor(s): {', '.join(sorted(unknown))}")
    sweeps = [sweep for sweep in SWEEPS if sweep.processor in selected]

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "one_factor_at_a_time": True,
            "pilot_points": 3,
            "pilot_repetitions": args.pilot_repetitions,
            "full_repetitions": args.full_repetitions,
            "warmups": args.warmups,
            "early_stop_rule": (
                "stop after pilot when max median minus min median is no greater than "
                "max(15% of the minimum, 0.20 ms CPU / 0.03 ms GPU)"
            ),
            "timed_region": "Processor call only; input generation and transform fitting excluded",
            "cuda_timing": "synchronized wall, CUDA event, enqueue, synchronization wait, peak allocated bytes",
        },
        "defaults": DEFAULTS,
        "hardware": _hardware(requested_devices),
        "processor_metadata": PROCESSOR_META,
        "sweeps": [],
    }
    total = len(sweeps) * len(requested_devices)
    completed = 0
    for device in requested_devices:
        for sweep in sweeps:
            completed += 1
            print(
                f"[{completed}/{total}] {device.type} {sweep.processor}.{sweep.operation} / {sweep.axis.name}",
                flush=True,
            )
            try:
                result = _run_sweep(
                    sweep,
                    device,
                    pilot_repetitions=args.pilot_repetitions,
                    full_repetitions=args.full_repetitions,
                    warmups=args.warmups,
                )
            except (
                NotImplementedError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                result = {
                    "processor": sweep.processor,
                    "operation": sweep.operation,
                    "stypes": list(sweep.stypes),
                    "axis": sweep.axis.name,
                    "device": device.type,
                    "status": "error",
                    "reason": f"{type(error).__name__}: {error}",
                    "points": [],
                }
            payload["sweeps"].append(result)
            gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not args.no_plots:
        _write_plots(payload, args.plots_dir)
    errors = [
        result for result in payload["sweeps"] if result["status"] == "error"
    ]
    print(f"wrote {args.output} ({len(errors)} errors)")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
