"""Benchmark TabPFN-3 and KumoTabular runtime stages."""

from __future__ import annotations

import argparse
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch

import sdm

Task = Literal["classification", "regression"]
PredictionMethod = Literal["predict", "predict_proba"]
TabPFNFitMode = Literal[
    "low_memory",
    "fit_preprocessors",
    "fit_with_cache",
    "batched",
]
TabPFNCachePrecision = Literal["auto", "int8", "fp8"]


@dataclass(frozen=True)
class RawData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray


@dataclass(frozen=True)
class BenchmarkConfig:
    task: Task
    device: torch.device
    num_train: int
    num_test: int
    num_features: int
    num_classes: int
    num_estimators: int
    seed: int
    warmup_runs: int
    fit_runs: int
    predict_runs: int
    predict_calls: int
    kumo_size: Literal["small", "large"]
    tabpfn_predict_method: PredictionMethod
    tabpfn_fit_mode: TabPFNFitMode
    tabpfn_keep_cache_on_device: bool
    tabpfn_kv_cache_precision: TabPFNCachePrecision | None


@dataclass
class Result:
    model: str
    stage: str
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    runs: int


class Adapter(Protocol):
    name: str

    def preprocess(self, raw: RawData) -> Any:
        pass

    def create_model(self) -> Any:
        pass

    def fit(self, model: Any, data: Any) -> None:
        pass

    def predict(self, model: Any, data: Any) -> Any:
        pass


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn: Callable[[], Any]) -> tuple[float, Any]:
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return time.perf_counter() - start, out


def _summarize(model: str, stage: str, times: list[float]) -> Result:
    values = np.asarray(times, dtype=np.float64)
    return Result(
        model=model,
        stage=stage,
        mean_s=float(values.mean()),
        std_s=float(values.std(ddof=0)),
        min_s=float(values.min()),
        max_s=float(values.max()),
        runs=len(times),
    )


def _generate_data(config: BenchmarkConfig) -> RawData:
    rng = np.random.default_rng(config.seed)
    x_train = rng.standard_normal(
        (config.num_train, config.num_features),
        dtype=np.float32,
    )
    x_test = rng.standard_normal(
        (config.num_test, config.num_features),
        dtype=np.float32,
    )

    if config.task == "regression":
        weights = rng.standard_normal(config.num_features, dtype=np.float32)
        y_train = x_train @ weights
        noise = rng.standard_normal(config.num_train, dtype=np.float32)
        y_train += 0.1 * noise
        return RawData(x_train=x_train, y_train=y_train, x_test=x_test)

    weights = rng.standard_normal(
        (config.num_features, config.num_classes),
        dtype=np.float32,
    )
    scores = x_train @ weights
    scores += 0.1 * rng.standard_normal(scores.shape, dtype=np.float32)
    y_train = scores.argmax(axis=1).astype(np.int64)
    return RawData(x_train=x_train, y_train=y_train, x_test=x_test)


def _maybe_device_kwargs(
    cls: type,
    device: torch.device,
) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    if "device" in parameters:
        return {"device": str(device)}
    return {}


def _supported_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
    parameters = inspect.signature(cls).parameters
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters and value is not None
    }


class TabPFNAdapter:
    name = "TabPFN-3"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def preprocess(
        self,
        raw: RawData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_dtype = np.float32 if self.config.task == "regression" else np.int64
        return (
            np.asarray(raw.x_train, dtype=np.float32),
            np.asarray(raw.y_train, dtype=y_dtype),
            np.asarray(raw.x_test, dtype=np.float32),
        )

    def create_model(self) -> Any:
        if self.config.task == "regression":
            from tabpfn import TabPFNRegressor  # noqa: PLC0415

            return TabPFNRegressor(
                **_maybe_device_kwargs(TabPFNRegressor, self.config.device),
                **_supported_kwargs(
                    TabPFNRegressor,
                    fit_mode=self.config.tabpfn_fit_mode,
                    keep_cache_on_device=(
                        self.config.tabpfn_keep_cache_on_device
                    ),
                    kv_cache_precision=self.config.tabpfn_kv_cache_precision,
                ),
            )

        from tabpfn import TabPFNClassifier  # noqa: PLC0415

        return TabPFNClassifier(
            **_maybe_device_kwargs(TabPFNClassifier, self.config.device),
            **_supported_kwargs(
                TabPFNClassifier,
                fit_mode=self.config.tabpfn_fit_mode,
                keep_cache_on_device=self.config.tabpfn_keep_cache_on_device,
                kv_cache_precision=self.config.tabpfn_kv_cache_precision,
            ),
        )

    def fit(
        self,
        model: Any,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x_train, y_train, _ = data
        model.fit(x_train, y_train)

    def predict(
        self,
        model: Any,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Any:
        _, _, x_test = data
        if (
            self.config.task == "classification"
            and self.config.tabpfn_predict_method == "predict_proba"
        ):
            return model.predict_proba(x_test)
        return model.predict(x_test)


class KumoAdapter:
    name = "KumoTabular"

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def preprocess(
        self,
        raw: RawData,
    ) -> tuple[sdm.TableTensor, sdm.TableTensor, sdm.TableTensor]:
        feature_columns = tuple(f"x_{i}" for i in range(raw.x_train.shape[1]))
        x_train = sdm.TableTensor(
            columns={sdm.Stype.numerical: feature_columns},
            numerical=torch.as_tensor(
                raw.x_train,
                device=self.config.device,
            ),
        )
        x_test = sdm.TableTensor(
            columns={sdm.Stype.numerical: feature_columns},
            numerical=torch.as_tensor(
                raw.x_test,
                device=self.config.device,
            ),
        )

        if self.config.task == "regression":
            y_train = sdm.TableTensor(
                columns={sdm.Stype.numerical: ("target",)},
                numerical=torch.as_tensor(
                    raw.y_train[:, None],
                    device=self.config.device,
                ),
            )
            return x_train, y_train, x_test

        code = torch.as_tensor(
            raw.y_train[:, None],
            device=self.config.device,
        )
        y_train = sdm.TableTensor(
            columns={sdm.Stype.categorical: ("target",)},
            categorical=sdm.CategoricalTensor.from_tensor(code),
        )
        return x_train, y_train, x_test

    def create_model(self) -> sdm.models.KumoTabular:
        return sdm.models.KumoTabular(
            task=self.config.task,
            size=self.config.kumo_size,
            device=self.config.device,
        )

    def fit(
        self,
        model: sdm.models.KumoTabular,
        data: tuple[sdm.TableTensor, sdm.TableTensor, sdm.TableTensor],
    ) -> None:
        x_train, y_train, _ = data
        generator = torch.Generator(device=self.config.device).manual_seed(
            self.config.seed,
        )
        with torch.amp.autocast(
            self.config.device.type,
            torch.float16,
            enabled=x_train.is_cuda,
        ):
            model.fit(
                x=x_train,
                y=y_train,
                num_estimators=self.config.num_estimators,
                generator=generator,
            )

    def predict(
        self,
        model: sdm.models.KumoTabular,
        data: tuple[sdm.TableTensor, sdm.TableTensor, sdm.TableTensor],
    ) -> sdm.TableTensor:
        _, _, x_test = data
        with torch.amp.autocast(
            self.config.device.type,
            torch.float16,
            enabled=x_test.is_cuda,
        ):
            return model.predict(x_test)


def _benchmark_adapter(
    adapter: Adapter,
    raw: RawData,
    config: BenchmarkConfig,
) -> tuple[list[Result], Any]:
    preprocess_times: list[float] = []
    data = None
    for _ in range(config.warmup_runs):
        _, data = _timed(config.device, lambda: adapter.preprocess(raw))
    for _ in range(config.fit_runs):
        elapsed, data = _timed(config.device, lambda: adapter.preprocess(raw))
        preprocess_times.append(elapsed)

    assert data is not None
    model = adapter.create_model()

    for _ in range(config.warmup_runs):
        adapter.fit(model, data)
        for _ in range(config.predict_calls):
            adapter.predict(model, data)
        _sync(config.device)

    fit_times: list[float] = []
    for _ in range(config.fit_runs):
        elapsed, _ = _timed(config.device, lambda: adapter.fit(model, data))
        fit_times.append(elapsed)

    predict_times: dict[int, list[float]] = {
        call: [] for call in range(1, config.predict_calls + 1)
    }
    for _ in range(config.predict_runs):
        adapter.fit(model, data)
        _sync(config.device)
        for call in predict_times:
            elapsed, _ = _timed(
                config.device,
                lambda: adapter.predict(model, data),
            )
            predict_times[call].append(elapsed)

    return (
        [
            _summarize(adapter.name, "preprocess", preprocess_times),
            _summarize(adapter.name, "fit", fit_times),
            *[
                _summarize(adapter.name, f"predict_{call}", times)
                for call, times in predict_times.items()
            ],
        ],
        model,
    )


def _print_results(results: list[Result]) -> None:
    headers = ("model", "stage", "mean_s", "std_s", "min_s", "max_s", "runs")
    rows = [
        (
            result.model,
            result.stage,
            f"{result.mean_s:.6f}",
            f"{result.std_s:.6f}",
            f"{result.min_s:.6f}",
            f"{result.max_s:.6f}",
            str(result.runs),
        )
        for result in results
    ]
    widths = [
        max(len(row[i]) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    print(
        "  ".join(
            value.ljust(width) for value, width in zip(headers, widths)
        )
    )
    for row in rows:
        print(
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths)
            )
        )


def _write_csv(path: Path, results: list[Result]) -> None:
    lines = ["model,stage,mean_s,std_s,min_s,max_s,runs"]
    lines.extend(
        (
            f"{result.model},{result.stage},{result.mean_s:.9f},"
            f"{result.std_s:.9f},{result.min_s:.9f},"
            f"{result.max_s:.9f},{result.runs}"
        )
        for result in results
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark API-level preprocessing, fit, and predict runtime for "
            "TabPFN-3 and KumoTabular on synthetic tabular data."
        ),
    )
    parser.add_argument(
        "--models",
        choices=("tabpfn3", "kumo"),
        nargs="+",
        default=("tabpfn3", "kumo"),
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-train", type=int, default=1024)
    parser.add_argument("--num-test", type=int, default=512)
    parser.add_argument("--num-features", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--fit-runs", type=int, default=3)
    parser.add_argument("--predict-runs", type=int, default=5)
    parser.add_argument("--predict-calls", type=int, default=2)
    parser.add_argument(
        "--kumo-size",
        choices=("small", "large"),
        default="large",
    )
    parser.add_argument(
        "--tabpfn-predict-method",
        choices=("predict", "predict_proba"),
        default="predict_proba",
    )
    parser.add_argument(
        "--tabpfn-fit-mode",
        choices=(
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            "batched",
        ),
        default="fit_with_cache",
    )
    parser.add_argument(
        "--tabpfn-keep-cache-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tabpfn-kv-cache-precision",
        choices=("auto", "int8", "fp8"),
        default=None,
    )
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_calls < 1:
        raise ValueError("Expected '--predict-calls' to be at least 1")

    device = torch.device(
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    config = BenchmarkConfig(
        task=args.task,
        device=device,
        num_train=args.num_train,
        num_test=args.num_test,
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_estimators=args.num_estimators,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        fit_runs=args.fit_runs,
        predict_runs=args.predict_runs,
        predict_calls=args.predict_calls,
        kumo_size=args.kumo_size,
        tabpfn_predict_method=args.tabpfn_predict_method,
        tabpfn_fit_mode=args.tabpfn_fit_mode,
        tabpfn_keep_cache_on_device=args.tabpfn_keep_cache_on_device,
        tabpfn_kv_cache_precision=args.tabpfn_kv_cache_precision,
    )
    raw = _generate_data(config)

    adapters: list[Adapter] = []
    if "tabpfn3" in args.models:
        adapters.append(TabPFNAdapter(config))
    if "kumo" in args.models:
        adapters.append(KumoAdapter(config))

    print(
        "Benchmarking "
        f"task={config.task}, device={config.device}, "
        f"train={config.num_train}, test={config.num_test}, "
        f"features={config.num_features}"
    )
    print(
        "Model initialization and checkpoint download are outside timed "
        "stages, except for lazy downloads triggered by a model's first fit."
    )
    print(
        "Each measured prediction run uses an untimed fresh fit followed by "
        f"{config.predict_calls} timed consecutive predict calls."
    )
    print(
        f"TabPFN fit_mode={config.tabpfn_fit_mode!r}, "
        "keep_cache_on_device="
        f"{config.tabpfn_keep_cache_on_device}, "
        f"kv_cache_precision={config.tabpfn_kv_cache_precision!r}."
    )

    results: list[Result] = []
    models: list[Any] = []
    try:
        for adapter in adapters:
            adapter_results, model = _benchmark_adapter(adapter, raw, config)
            results.extend(adapter_results)
            models.append(model)
            _print_results(adapter_results)
    finally:
        for model in models:
            clear = getattr(model, "clear", None)
            if callable(clear):
                clear()

    _print_results(results)
    if args.csv is not None:
        _write_csv(args.csv, results)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
