"""Benchmark categorical ingestion constructors.

This script compares equivalent pandas, Arrow, and cuDF inputs for
``CategoricalTensor``. It times constructor work only; input materialization is
done before each benchmark case.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata as metadata
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdm import CategoricalTensor
from sdm.tensor.categorical import _category_tensor


@dataclass(frozen=True)
class _Case:
    kind: str
    rows: int
    cardinality: int
    null_rate: float


@dataclass(frozen=True)
class _Inputs:
    pandas: pd.Series
    arrow: pa.Array
    cudf: Any


@dataclass(frozen=True)
class _Timing:
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class _Result:
    case: _Case
    pandas: _Timing
    arrow: _Timing
    cudf: _Timing


def _parse_ints(value: str) -> list[int]:
    return [int(item.replace("_", "")) for item in value.split(",")]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return "not-installed"


def _environment_markdown(cudf: Any, cupy: Any) -> str:
    lines = [
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- torch CUDA runtime: `{torch.version.cuda}`",
        f"- torch.cuda.is_available(): `{torch.cuda.is_available()}`",
        (
            f"- cuDF: `{getattr(cudf, '__version__', 'unknown')}` "
            f"(distribution "
            f"`{_package_version('cudf', 'cudf-cu13', 'cudf-cu12')}`)"
        ),
        (
            f"- CuPy: `{getattr(cupy, '__version__', 'unknown')}` "
            f"(distribution "
            f"`{_package_version('cupy', 'cupy-cuda13x', 'cupy-cuda12x')}`)"
        ),
        f"- pandas: `{pd.__version__}`",
        f"- pyarrow: `{pa.__version__}`",
        f"- numpy: `{np.__version__}`",
        "",
    ]
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        lines.append(
            "- GPU "
            f"{index}: `{props.name}`, compute capability "
            f"`{props.major}.{props.minor}`, memory "
            f"`{props.total_memory / 1024**3:.2f} GiB`"
        )
    lines.append("")
    return "\n".join(lines)


def _build_inputs(case: _Case, *, seed: int, cudf: Any) -> _Inputs:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, case.cardinality, size=case.rows, dtype=np.int64)
    null_mask = rng.random(case.rows) < case.null_rate

    if case.kind == "numeric":
        values = (codes * 3 + 7).astype(np.int64, copy=False)
        pandas_series = pd.Series(values, dtype="Int64")
        if null_mask.any():
            pandas_series[null_mask] = pd.NA
        arrow_array = pa.array(values, mask=null_mask, type=pa.int64())
    elif case.kind == "string":
        labels = np.array(
            [f"cat_{index:06d}" for index in range(case.cardinality)],
            dtype=object,
        )
        values = labels[codes]
        if null_mask.any():
            values = values.copy()
            values[null_mask] = None
        pandas_series = pd.Series(values, dtype="object")
        arrow_array = pa.array(values, type=pa.string())
    else:
        raise ValueError(f"Unsupported kind: {case.kind}")

    return _Inputs(
        pandas=pandas_series,
        arrow=arrow_array,
        cudf=cudf.Series(pandas_series),
    )


def _time_callable(
    fn: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
    sync_cuda: bool,
) -> _Timing:
    for _ in range(warmups):
        out = fn()
        if sync_cuda:
            _sync_cuda()
        del out

    gc.collect()
    samples: list[float] = []
    for _ in range(repeats):
        if sync_cuda:
            _sync_cuda()
        start = time.perf_counter()
        out = fn()
        if sync_cuda:
            _sync_cuda()
        samples.append((time.perf_counter() - start) * 1000)
        del out

    return _Timing(
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def _profile_cudf(
    series: Any,
    *,
    repeats: int,
) -> dict[str, float]:
    stages = {
        "factorize": [],
        "astype": [],
        "dlpack_codes": [],
        "category_materialization": [],
        "wrap": [],
    }

    for _ in range(repeats):
        gc.collect()
        _sync_cuda()

        start = time.perf_counter()
        codes, categories = series.factorize(
            sort=False,
            use_na_sentinel=True,
        )
        _sync_cuda()
        stages["factorize"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        codes = codes.astype("int32", copy=False)
        _sync_cuda()
        stages["astype"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        data = torch.from_dlpack(codes).unsqueeze(-1)
        _sync_cuda()
        stages["dlpack_codes"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        if len(categories) == 0:
            category = torch.empty(0, dtype=torch.int64)
        elif getattr(categories.dtype, "kind", None) == "O":
            category = _category_tensor(
                categories.to_pandas().tolist(), device=None
            )
        else:
            category = torch.from_dlpack(categories.to_cupy())
        _sync_cuda()
        stages["category_materialization"].append(
            (time.perf_counter() - start) * 1000
        )

        start = time.perf_counter()
        out = CategoricalTensor(data=data, categories=(category,))
        _sync_cuda()
        stages["wrap"].append((time.perf_counter() - start) * 1000)
        del out

    return {name: statistics.median(values) for name, values in stages.items()}


def _format_ms(timing: _Timing) -> str:
    return f"{timing.median_ms:.2f}"


def _results_markdown(results: Iterable[_Result]) -> str:
    lines = [
        "## Constructor Timings",
        "",
        "Median milliseconds; lower is better. `CPU best / cuDF` compares "
        "cuDF against the faster of pandas and Arrow.",
        "",
        "| kind | rows | cardinality | null rate | pandas ms | Arrow ms | "
        "cuDF ms | CPU best / cuDF | fastest |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        cpu_best = min(result.pandas.median_ms, result.arrow.median_ms)
        ratio = cpu_best / result.cudf.median_ms
        timings = {
            "pandas": result.pandas.median_ms,
            "Arrow": result.arrow.median_ms,
            "cuDF": result.cudf.median_ms,
        }
        fastest = min(timings, key=timings.__getitem__)
        lines.append(
            f"| {result.case.kind} | {result.case.rows:,} | "
            f"{result.case.cardinality:,} | {result.case.null_rate:.2%} | "
            f"{_format_ms(result.pandas)} | {_format_ms(result.arrow)} | "
            f"{_format_ms(result.cudf)} | {ratio:.2f}x | {fastest} |"
        )
    lines.append("")
    return "\n".join(lines)


def _profiles_markdown(
    profiles: dict[_Case, dict[str, float]],
) -> str:
    if not profiles:
        return (
            "## cuDF Slow-Case Profiles\n\n"
            "cuDF was faster than the best CPU constructor in every case, so "
            "no slow-case profile was collected.\n"
        )

    lines = [
        "## cuDF Slow-Case Profiles",
        "",
        "Median milliseconds by stage for cases where cuDF was slower than "
        "the faster CPU constructor.",
        "",
        "| kind | rows | cardinality | null rate | factorize | astype | "
        "DLPack codes | category materialization | wrap | profiled total |",
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |"
        ),
    ]
    for case, profile in profiles.items():
        total = sum(profile.values())
        lines.append(
            f"| {case.kind} | {case.rows:,} | {case.cardinality:,} | "
            f"{case.null_rate:.2%} | {profile['factorize']:.2f} | "
            f"{profile['astype']:.2f} | {profile['dlpack_codes']:.2f} | "
            f"{profile['category_materialization']:.2f} | "
            f"{profile['wrap']:.2f} | {total:.2f} |"
        )
    lines.extend(
        [
            "",
            "Reliability hypothesis: slow string cases are expected when "
            "category materialization dominates because the current cuDF "
            "constructor converts string categories through "
            "`categories.to_pandas().tolist()` before building "
            "`StringTensor`. Small numeric cases can also be slower when "
            "GPU factorization launch overhead outweighs CPU dictionary "
            "encoding. Larger numeric cases should be the most reliable "
            "place to see cuDF's zero-copy DLPack advantage.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available according to PyTorch")
    _sync_cuda()


def _validate_result(tensor: CategoricalTensor, rows: int) -> None:
    if tensor.shape != (rows, 1):
        raise RuntimeError(
            f"Unexpected tensor shape: {tuple(tensor.shape)} for {rows} rows"
        )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark categorical ingestion constructors."
    )
    parser.add_argument("--rows", default="100_000,1_000_000")
    parser.add_argument("--cardinalities", default="100,10_000")
    parser.add_argument("--null-rates", default="0,0.05")
    parser.add_argument("--kinds", default="numeric,string")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cupy = importlib.import_module("cupy")
    cudf = importlib.import_module("cudf")

    _validate_cuda()

    cases = [
        _Case(
            kind=kind, rows=rows, cardinality=cardinality, null_rate=null_rate
        )
        for kind in args.kinds.split(",")
        for rows in _parse_ints(args.rows)
        for cardinality in _parse_ints(args.cardinalities)
        for null_rate in _parse_floats(args.null_rates)
    ]

    results: list[_Result] = []
    profiles: dict[_Case, dict[str, float]] = {}
    for index, case in enumerate(cases):
        inputs = _build_inputs(case, seed=args.seed + index, cudf=cudf)
        _sync_cuda()

        pandas_timing = _time_callable(
            lambda series=inputs.pandas: CategoricalTensor.from_pandas(series),
            repeats=args.repeats,
            warmups=args.warmups,
            sync_cuda=False,
        )
        arrow_timing = _time_callable(
            lambda array=inputs.arrow: CategoricalTensor.from_arrow(array),
            repeats=args.repeats,
            warmups=args.warmups,
            sync_cuda=False,
        )
        cudf_timing = _time_callable(
            lambda series=inputs.cudf: CategoricalTensor.from_cudf(series),
            repeats=args.repeats,
            warmups=args.warmups,
            sync_cuda=True,
        )

        pandas_tensor = CategoricalTensor.from_pandas(inputs.pandas)
        arrow_tensor = CategoricalTensor.from_arrow(inputs.arrow)
        cudf_tensor = CategoricalTensor.from_cudf(inputs.cudf)
        _sync_cuda()
        _validate_result(pandas_tensor, case.rows)
        _validate_result(arrow_tensor, case.rows)
        _validate_result(cudf_tensor, case.rows)

        result = _Result(
            case=case,
            pandas=pandas_timing,
            arrow=arrow_timing,
            cudf=cudf_timing,
        )
        results.append(result)

        cpu_best = min(pandas_timing.median_ms, arrow_timing.median_ms)
        if cudf_timing.median_ms > cpu_best:
            profiles[case] = _profile_cudf(
                inputs.cudf,
                repeats=args.profile_repeats,
            )

        sys.stdout.write(
            f"{case.kind} rows={case.rows:,} card={case.cardinality:,} "
            f"null={case.null_rate:.2%}: "
            f"pandas={pandas_timing.median_ms:.2f} ms, "
            f"arrow={arrow_timing.median_ms:.2f} ms, "
            f"cudf={cudf_timing.median_ms:.2f} ms\n"
        )

    markdown = "\n".join(
        [
            _environment_markdown(cudf, cupy),
            _results_markdown(results),
            _profiles_markdown(profiles),
        ]
    )
    sys.stdout.write(f"\n{markdown}\n")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    _main()
