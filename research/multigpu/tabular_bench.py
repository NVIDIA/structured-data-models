"""Measure KumoTabular inference with fixed data, query boundaries, and seeds.

Run one configuration per fresh process. Input NPZ contains x_context,
y_context, x_query, y_query. Validation targets are used only after prediction.
Adapters implement factory(args, replicas) -> object exposing fit/predict.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular


def synchronize(devices: list[str]) -> None:
    """Wait for all participating devices at measurement boundaries."""
    for device in devices:
        torch.cuda.synchronize(device)


def memory(devices: list[str]) -> list[dict[str, Any]]:
    """Capture current and peak allocator counters per device."""
    return [
        {
            "device": device,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        for device in devices
    ]


def cache_memory(model: Any) -> dict[str, int]:
    """Count unique cache tensor storages, grouped by residency."""
    result: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    caches = []
    pending = [model]
    while pending:
        current = pending.pop()
        caches.append(getattr(current, "_cache", None))
        caches.extend(getattr(current, "_caches", []))
        pending.extend(getattr(current, "replicas", []))
    for cache in caches:
        if isinstance(cache, Cache):
            for tensor in cache._tensors():
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    result[key[0]] = result.get(key[0], 0) + storage.nbytes()
    return result


def quality(
    task: str, target: np.ndarray, prediction: np.ndarray, columns: list[str]
) -> dict[str, float]:
    """Score held-out predictions using their explicit output class order."""
    target = target.reshape(-1)
    if task == "regression":
        median = prediction[:, prediction.shape[1] // 2]
        return {
            "rmse": float(mean_squared_error(target, median) ** 0.5),
            "mae": float(mean_absolute_error(target, median)),
            "quantile_monotonicity_violation_fraction": float(
                (np.diff(prediction, axis=1) < 0).mean()
            ),
        }
    classes = np.array([int(column) for column in columns])
    result = {
        "log_loss": float(log_loss(target, prediction, labels=classes)),
        "accuracy": float(
            accuracy_score(target, classes[prediction.argmax(axis=1)])
        ),
    }
    if prediction.shape[1] == 2:
        result["auroc"] = float(
            roc_auc_score(target == classes[1], prediction[:, 1])
        )
    else:
        result["auroc_ovr"] = float(
            roc_auc_score(target, prediction, multi_class="ovr")
        )
    return result


def timed(call: Any, devices: list[str]) -> tuple[Any, float]:
    """Measure synchronized wall time including all participating devices."""
    synchronize(devices)
    start = time.perf_counter()
    result = call()
    synchronize(devices)
    return result, time.perf_counter() - start


def main() -> None:
    """Run one configuration and retain raw measurements and predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit", help="Git revision for an archived source tree"
    )
    parser.add_argument(
        "--task", choices=["classification", "regression"], required=True
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="large"
    )
    parser.add_argument(
        "--mode", choices=["native", "ensemble", "adapter"], default="native"
    )
    parser.add_argument(
        "--adapter", help="module:factory for another parallel implementation"
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--replicas",
        type=int,
        help="Model copies to load; defaults to GPU count",
    )
    parser.add_argument(
        "--placement", choices=["stage", "layers"], default="layers"
    )
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--estimator-batch-size", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--nvtx", action="store_true", help="Annotate separate Nsight runs"
    )
    parser.add_argument(
        "--query-residency", choices=["cpu", "gpu"], default="gpu"
    )
    args = parser.parse_args()

    def phase(name: str) -> Any:
        return (
            torch.cuda.nvtx.range(name)
            if args.nvtx
            else contextlib.nullcontext()
        )

    args.output.mkdir(parents=True, exist_ok=False)
    devices = [f"cuda:{i}" for i in range(args.gpus)]
    if args.mode == "native" and args.gpus != 1:
        raise ValueError("The native baseline uses one GPU")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    report: dict[str, Any] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runtime": {
            "pid": os.getpid(),
            "timezone": list(time.tzname),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "host": platform.node(),
            "gpu_names": [
                torch.cuda.get_device_name(device) for device in devices
            ],
            "git_revision": args.source_commit
            or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    }
    telemetry = (args.output / "nvidia-smi.csv").open("w")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw",
            "--format=csv",
            "--loop-ms=200",
        ],
        stdout=telemetry,
        stderr=subprocess.DEVNULL,
    )
    try:
        data_start = time.perf_counter()
        if args.data.is_dir():
            data = {
                key: np.load(args.data / f"{name}.npy", mmap_mode="r")
                for key, name in {
                    "x_context": "x_train",
                    "y_context": "y_train",
                    "x_query": "x_val",
                    "y_query": "y_val",
                    "query_ids": "val_ids",
                }.items()
            }
        else:
            data = np.load(args.data, allow_pickle=False)
        x_context = torch.from_numpy(
            data["x_context"][: args.context].copy()
        ).float()
        y_context = torch.from_numpy(
            data["y_context"][: args.context].copy()
        ).reshape(-1, 1)
        x_query = torch.from_numpy(
            data["x_query"][: args.queries].copy()
        ).float()
        target = data["y_query"][: args.queries].copy()
        query_ids = (
            data["query_ids"][: args.queries].copy()
            if "query_ids" in data
            else np.arange(args.queries)
        )
        np.save(args.output / "query_ids.npy", query_ids)
        np.save(args.output / "targets.npy", target)
        if (
            x_context.shape[0] != args.context
            or x_query.shape[0] != args.queries
        ):
            raise ValueError("Requested rows exceed the fixed dataset split")
        report["data_load_s"] = time.perf_counter() - data_start
        report["input_shape"] = {
            "context": list(x_context.shape),
            "query": list(x_query.shape),
        }
        report["input_sha256"] = {
            key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in [
                ("x_context", x_context),
                ("y_context", y_context),
                ("x_query", x_query),
            ]
        }

        def transfer() -> tuple[TableTensor, TableTensor, TableTensor]:
            context = TableTensor.from_tensor(x_context.to(devices[0]))
            query = TableTensor.from_tensor(
                x_query.to(
                    devices[0] if args.query_residency == "gpu" else "cpu"
                )
            )
            if args.task == "classification":
                labels = TableTensor(
                    columns={Stype.categorical: ["target"]},
                    categorical=CategoricalTensor.from_tensor(
                        y_context.long().to(devices[0])
                    ),
                )
            else:
                labels = TableTensor.from_tensor(
                    y_context.float().to(devices[0])
                )
            return context, labels, query

        (context, labels, query), report["input_transfer_s"] = timed(
            transfer, devices
        )

        def load() -> Any:
            replicas = [
                KumoTabular(task=args.task, size=args.size, device=device)
                for device in devices[: args.replicas or args.gpus]
            ]
            if args.mode == "native":
                return replicas[0]
            if args.mode == "ensemble":
                from sdm.models import EnsembleParallel  # noqa: PLC0415

                return EnsembleParallel(replicas)
            module, function = args.adapter.split(":")
            return getattr(importlib.import_module(module), function)(
                args, replicas
            )

        report["load_start_unix_s"] = time.time()
        with phase("model_load"):
            model, report["load_s"] = timed(load, devices)
        report["load_end_unix_s"] = time.time()
        print(  # noqa: T201
            json.dumps({"phase": "loaded", "seconds": report["load_s"]}),
            flush=True,
        )
        report["memory_after_load"] = memory(devices)
        if callable(getattr(model, "memory", None)):
            report["worker_memory_after_load"] = model.memory()

        def dtype_context() -> torch.autocast:
            return torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=args.precision == "bfloat16",
            )

        generator = torch.Generator(device=devices[0]).manual_seed(args.seed)
        fit_kwargs: dict[str, Any] = {
            "num_estimators": args.estimators,
            "generator": generator,
        }
        if args.mode == "native":
            fit_kwargs["estimator_batch_size"] = args.estimator_batch_size
        elif (
            args.mode == "ensemble"
            or model.__class__.__name__ == "EnsembleParallel"
        ):
            fit_kwargs["member_seed"] = args.seed
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        report["fit_start_unix_s"] = time.time()
        with phase("context_fit"), torch.inference_mode(), dtype_context():
            _, report["fit_s"] = timed(
                lambda: model.fit(context, labels, **fit_kwargs), devices
            )
        report["fit_end_unix_s"] = time.time()
        print(  # noqa: T201
            json.dumps({"phase": "fitted", "seconds": report["fit_s"]}),
            flush=True,
        )
        report["memory_after_fit"] = memory(devices)
        if callable(getattr(model, "memory", None)):
            report["worker_memory_after_fit"] = model.memory()
        report["cache_storage_bytes_after_fit"] = cache_memory(model)
        if hasattr(model, "cache_bytes"):
            report["executor_cache_bytes"] = list(model.cache_bytes)
        if hasattr(model, "member_seeds"):
            report["member_seeds"] = list(model.member_seeds)
        if hasattr(model, "compaction_s"):
            report["compaction_s"] = model.compaction_s
            report["compaction_storage_bytes"] = model.compaction_storage_bytes
        batches = list(query.split(args.batch_size))

        def predict_batch(batch: TableTensor) -> TableTensor:
            return model.predict(batch.to(devices[0]))

        def predict_pass() -> tuple[torch.Tensor, list[float]]:
            if hasattr(model, "predict_batches"):
                outputs = model.predict_batches(batches)
                report["prediction_columns"] = list(
                    outputs[0].columns[Stype.numerical]
                )
                return torch.cat([output.numerical for output in outputs]), []
            outputs, times = [], []
            for batch in batches:
                output, elapsed = timed(
                    lambda batch=batch: predict_batch(batch), devices
                )
                report["prediction_columns"] = list(
                    output.columns[Stype.numerical]
                )
                outputs.append(output.numerical.float().cpu())
                times.append(elapsed)
            return torch.cat(outputs), times

        with torch.inference_mode(), dtype_context():
            warmup_start = time.perf_counter()
            for _ in range(args.warmups):
                if hasattr(model, "predict_batches"):
                    timed(
                        lambda: model.predict_batches(batches[: args.gpus]),
                        devices,
                    )
                else:
                    timed(lambda: predict_batch(batches[0]), devices)
            report["warmup_s"] = time.perf_counter() - warmup_start
            if hasattr(model, "graph_count"):
                report["graph_count_after_warmup"] = model.graph_count
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            if callable(getattr(model, "memory", None)):
                model.memory(reset_peak=True)
            repeats, batches_s, predictions = [], [], []
            report["prediction_windows_unix_s"] = []
            for _ in range(args.repeats):
                pass_start = time.time()
                with phase("prediction_pass"):
                    (prediction, batch_times), elapsed = timed(
                        predict_pass, devices
                    )
                report["prediction_windows_unix_s"].append(
                    [pass_start, time.time()]
                )
                repeats.append(elapsed)
                print(  # noqa: T201
                    json.dumps(
                        {"phase": "prediction_pass", "seconds": elapsed}
                    ),
                    flush=True,
                )
                batches_s.append(batch_times)
                predictions.append(prediction.detach().float().cpu().numpy())
            report["memory_after_prediction"] = memory(devices)
            if hasattr(model, "graph_count"):
                report["graph_count_after_prediction"] = model.graph_count
                report["graph_capture_s"] = model.graph_capture_s
                report["graph_capture_events"] = list(model.capture_events)
                report["capture_occurred_during_timing"] = (
                    model.graph_count != report["graph_count_after_warmup"]
                )
            if callable(getattr(model, "memory", None)):
                report["worker_memory_after_prediction"] = model.memory()
            report["predict_repeats_s"] = repeats
            report["batch_times_s"] = batches_s
            report["throughput_rows_s"] = [
                args.queries / elapsed for elapsed in repeats
            ]
            if batches_s[0]:
                report["batch_p50_s"] = float(np.median(batches_s))
                report["batch_p95_s"] = float(np.percentile(batches_s, 95))
            report["repeat_max_abs_difference"] = max(
                float(np.max(np.abs(predictions[0] - item)))
                for item in predictions
            )
            report["quality"] = quality(
                args.task, target, predictions[0], report["prediction_columns"]
            )
            report["prediction_sha256"] = hashlib.sha256(
                predictions[0].tobytes()
            ).hexdigest()
            np.save(args.output / "predictions.npy", predictions[0])
            if args.profile:
                with (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        profile_memory=True,
                    ) as profiler,
                    torch.profiler.record_function(
                        "profiled_prediction_batch"
                    ),
                ):
                    if hasattr(model, "predict_batches"):
                        timed(
                            lambda: model.predict_batches(
                                batches[: args.gpus]
                            ),
                            devices,
                        )
                    else:
                        timed(lambda: predict_batch(batches[0]), devices)
                profiler.export_chrome_trace(str(args.output / "trace.json"))
                (args.output / "profile.txt").write_text(
                    profiler.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=60
                    )
                )
        report["status"] = "complete"
    except Exception as error:
        report.update(status="error", error=repr(error))
        with contextlib.suppress(RuntimeError):
            report["memory_at_failure"] = memory(devices)
        raise
    finally:
        if "model" in locals() and hasattr(model, "close"):
            model.close()
        monitor.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            monitor.wait(timeout=5)
        telemetry.close()
        report["max_cpu_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        (args.output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
