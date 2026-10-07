# ruff: noqa: D103, PLC0415, T201, TID253
"""Native relational sampling and repeated multi-GPU inference benchmark.

Prepare once, then run every placement against identical sampled graphs.
Validation targets are saved separately and opened only after prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

import sdm


def digest_table(table: sdm.TableTensor) -> str:
    frame = table.to_pandas()
    values = pd.util.hash_pandas_object(frame, index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def graph_identity(sample: Any) -> dict[str, Any]:
    return {
        "task": digest_table(sample.task_table),
        "tables": {
            name: {"rows": len(table), "sha256": digest_table(table)}
            for name, table in sample.related_tables.tables.items()
        },
        "relationships": repr(sample.related_tables.relationships),
        "task_links": repr(sample.related_tables.task_links),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    tables, metadata = {}, {}
    for path in sorted((args.raw / "db").glob("*.parquet")):
        arrow = pq.read_table(path)
        meta = arrow.schema.metadata or {}
        metadata[path.stem] = {
            key: json.loads(meta[key.encode()])
            for key in ["pkey_col", "fkey_col_to_pkey_table", "time_col"]
        }
        frame = arrow.to_pandas()
        keys = metadata[path.stem]["fkey_col_to_pkey_table"]
        overrides = dict.fromkeys(keys, "id")
        if metadata[path.stem]["pkey_col"] is not None:
            overrides[metadata[path.stem]["pkey_col"]] = "id"
        stypes = sdm.infer_stypes(
            frame.head(10_000),
            overrides=overrides,
            text="drop",
            unsupported="drop",
        )
        tables[path.stem] = sdm.TableTensor.from_pandas(frame, stypes)
    db_load_s = time.perf_counter() - start
    data = sdm.RelationalData(
        tables=tables,
        relationships=[
            {
                "left_table": name,
                "left_column": column,
                "right_table": right,
                "right_column": metadata[right]["pkey_col"],
            }
            for name, meta in metadata.items()
            for column, right in meta["fkey_col_to_pkey_table"].items()
        ],
    )
    start = time.perf_counter()
    sampler = data.sampler(
        time_columns={
            name: meta["time_col"]
            for name, meta in metadata.items()
            if meta["time_col"] is not None
        }
    )
    sampler_build_s = time.perf_counter() - start
    task_root = args.raw / "tasks" / args.task
    train = pd.read_parquet(task_root / "train.parquet")
    val = (
        pd.read_parquet(task_root / "val.parquet").iloc[: args.queries].copy()
    )
    selected = np.load(args.raw / f"{args.task}-train-ids.npy")[: args.context]
    context_frame = train.iloc[selected].copy()
    stypes = {args.entity: "id", args.time: "datetime"}
    target_stype = (
        "categorical" if args.problem == "classification" else "numerical"
    )
    context = sdm.TableTensor.from_pandas(
        context_frame[[args.entity, args.time, args.target]],
        {**stypes, args.target: target_stype},
    )
    query = sdm.TableTensor.from_pandas(val[[args.entity, args.time]], stypes)
    labels = sdm.TableTensor.from_pandas(
        val[[args.target]], {args.target: target_stype}
    )
    kwargs = {
        "task_link": {
            "task_column": args.entity,
            "table": args.entity_table,
            "table_column": metadata[args.entity_table]["pkey_col"],
        },
        "num_neighbors": args.neighbors,
        "task_time_column": args.time,
        "temporal_strategy": "last",
    }
    # Seed only sampling; subsequent model member RNG is independent.
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    sampled_context = sampler(context, **kwargs)
    context_sample_s = time.perf_counter() - start
    query_batches, sample_times, identities = [], [], []
    for batch in query.split(args.batch_size):
        start = time.perf_counter()
        sample = sampler(batch, **kwargs)
        sample_times.append(time.perf_counter() - start)
        query_batches.append(sample)
        identities.append(graph_identity(sample))
    manifest = {
        "dataset": args.raw.name,
        "task": args.task,
        "problem": args.problem,
        "target": args.target,
        "context": len(context),
        "queries": len(query),
        "batch_size": args.batch_size,
        "neighbors": args.neighbors,
        "temporal_strategy": "last",
        "seed": args.seed,
        "lag_features": 0,
        "context_row_indices_sha256": hashlib.sha256(
            selected.tobytes()
        ).hexdigest(),
        "query_row_indices_sha256": hashlib.sha256(
            np.arange(len(val)).tobytes()
        ).hexdigest(),
        "context_graph": graph_identity(sampled_context),
        "query_graphs": identities,
        "db_load_s": db_load_s,
        "sampler_build_s": sampler_build_s,
        "context_sample_s": context_sample_s,
        "query_sample_s": sample_times,
        "db_rows": {name: len(table) for name, table in tables.items()},
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    torch.save(
        {"context": sampled_context, "queries": query_batches},
        args.output / "graphs.pt",
    )
    torch.save(labels, args.output / "validation-labels.pt")
    np.save(args.output / "context-row-indices.npy", selected)
    write_json(args.output / "workload.json", manifest)
    print(
        json.dumps(
            {
                "prepared": str(args.output),
                "context": len(context),
                "queries": len(query),
            }
        ),
        flush=True,
    )


def synchronize(devices: list[str]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def memory(devices: list[str]) -> dict[str, Any]:
    return {
        device: {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
        for device in devices
    }


def score(
    pred: sdm.TableTensor, labels: sdm.TableTensor, problem: str
) -> dict[str, float]:
    if problem == "regression":
        values = pred["q500"].numerical.squeeze(-1).numpy()
        target = labels.numerical.squeeze(-1).numpy()
        quantiles = pred.numerical.numpy()
        return {
            "mae": mean_absolute_error(target, values),
            "rmse": mean_squared_error(target, values) ** 0.5,
            "nonfinite_predictions": int((~np.isfinite(quantiles)).sum()),
            "quantile_crossing_fraction": float(
                (np.diff(quantiles, axis=-1) < 0).mean()
            ),
        }
    probabilities, indices = sdm.evaluation.to_class_indices(pred, labels)
    probabilities, indices = probabilities.numpy(), indices.numpy()
    result = {
        "log_loss": log_loss(
            indices, probabilities, labels=np.arange(probabilities.shape[1])
        ),
        "accuracy": float((probabilities.argmax(-1) == indices).mean()),
    }
    if probabilities.shape[1] == 2:
        result["auroc"] = roc_auc_score(indices, probabilities[:, 1])
    return result


def run(args: argparse.Namespace) -> None:
    if args.mode == "native" and args.gpus != 1:
        raise ValueError("Native baseline uses exactly one GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = [f"cuda:{i}" for i in range(args.gpus)]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    workload = json.loads((args.workload / "workload.json").read_text())
    graphs = torch.load(args.workload / "graphs.pt", weights_only=False)
    context, batches = graphs["context"], graphs["queries"]
    task = workload["problem"]
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    stats: dict[str, Any] = {
        "args": vars(args),
        "workload": workload,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(device) for device in devices],
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    telemetry = (args.output / "nvidia-smi.csv").open("w")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv",
            "--loop-ms=200",
        ],
        stdout=telemetry,
        stderr=subprocess.DEVNULL,
    )
    try:
        synchronize(devices)
        start = time.perf_counter()
        replicas = [
            sdm.models.KumoRelational(task=task, device=device)
            for device in devices
        ]
        synchronize(devices)
        stats["load_s"] = time.perf_counter() - start
        stats["memory_after_load"] = memory(devices)
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        if args.mode == "ensemble":
            from sdm.models.ensemble_parallel import EnsembleParallel

            model = EnsembleParallel(replicas)
        else:
            model = replicas[0]
        x = context.task_table.drop_columns(workload["target"])
        y = context.task_table[workload["target"]]
        synchronize(devices)
        start = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ),
        ):
            if args.mode == "ensemble":
                model.fit(
                    x,
                    y,
                    context.related_tables,
                    num_estimators=args.estimators,
                    member_seed=args.seed,
                    generator=torch.Generator().manual_seed(args.seed),
                    num_hops=2,
                )
            else:
                for replica, device in zip(replicas, devices, strict=True):
                    replica.fit(
                        x.to(device),
                        y.to(device),
                        context.related_tables.to(device),
                        num_estimators=args.estimators,
                        generator=torch.Generator(device).manual_seed(
                            args.seed
                        ),
                        num_hops=2,
                    )
        synchronize(devices)
        stats["fit_s"] = time.perf_counter() - start
        stats["memory_after_fit"] = memory(devices)
        stats["native_cache_bytes"] = [
            replica._cache.size() if replica._cache is not None else 0
            for replica in replicas
        ]
        if args.mode == "ensemble":
            stats["ensemble_cache_bytes_per_gpu"] = model.cache_bytes
            stats["member_seeds"] = model.member_seeds
        query_executor, query_batches = None, None
        if args.mode == "data":
            from research.multigpu.query_parallel import (
                QueryBatch,
                QueryParallel,
            )

            query_executor = QueryParallel(
                replicas, devices=devices, dtype=dtype
            )
            offset = 0
            query_batches = []
            for batch in batches:
                rows = len(batch.task_table)
                query_batches.append(
                    QueryBatch(
                        tuple(range(offset, offset + rows)),
                        batch.task_table,
                        batch.related_tables,
                    )
                )
                offset += rows

        def predict() -> tuple[list[sdm.TableTensor], list[float]]:
            if query_executor is not None:
                results = query_executor.predict(query_batches)
                return [result.prediction for result in results], [
                    result.seconds for result in results
                ]
            predictions, durations = [], []
            for batch in batches:
                synchronize(devices)
                start = time.perf_counter()
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        "cuda", dtype=dtype, enabled=dtype != torch.float32
                    ),
                ):
                    if args.mode == "ensemble":
                        pred = model.predict(
                            batch.task_table, batch.related_tables
                        )
                    else:
                        pred = model.predict(*batch.to(devices[0]))
                    predictions.append(pred.cpu())
                synchronize(devices)
                durations.append(time.perf_counter() - start)
            return predictions, durations

        start = time.perf_counter()
        for _ in range(args.warmups):
            predict()
        synchronize(devices)
        stats["warmup_s"] = time.perf_counter() - start
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        elapsed, batch_times, arrays = [], [], []
        for _ in range(args.repeats):
            synchronize(devices)
            start = time.perf_counter()
            predictions, durations = predict()
            synchronize(devices)
            elapsed.append(time.perf_counter() - start)
            batch_times.append(durations)
            pred = torch.cat(predictions, dim=0)
            arrays.append(pred.numerical.numpy().copy())
        stats["predict_repeats_s"] = elapsed
        stats["batch_times_s"] = batch_times
        stats["batch_times_kind"] = (
            "worker_service_time_excluding_queue_delay"
            if args.mode == "data"
            else "synchronized_end_to_end_batch_latency"
        )
        stats["rows_per_s"] = [
            workload["queries"] / seconds for seconds in elapsed
        ]
        stats["memory_prediction"] = memory(devices)
        stats["repeat_max_abs_difference"] = [
            float(np.max(np.abs(array - arrays[0]))) for array in arrays
        ]
        stats["prediction_sha256"] = hashlib.sha256(
            arrays[0].tobytes()
        ).hexdigest()
        stats["prediction_columns"] = list(pred.columns[sdm.Stype.numerical])
        stats["max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        np.save(args.output / "predictions.npy", arrays[0])
        torch.save(pred, args.output / "predictions.pt")
        if args.profile:
            handles = []
            scopes = {}

            def begin(module: Any, inputs: Any) -> None:
                scope = torch.profiler.record_function(scopes[module][0])
                scope.__enter__()
                scopes[module][1].append(scope)

            def end(module: Any, inputs: Any, output: Any) -> None:
                scopes[module][1].pop().__exit__(None, None, None)

            for index, replica in enumerate(replicas):
                for core in replica.models.values():
                    for name in ["row_embedding", "gnn", "icl_block"]:
                        module = getattr(core, name)
                        scopes[module] = (f"gpu{index}/{name}", [])
                        handles.append(module.register_forward_pre_hook(begin))
                        handles.append(module.register_forward_hook(end))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                predict()
            for handle in handles:
                handle.remove()
            profiler.export_chrome_trace(str(args.output / "trace.json"))
            (args.output / "profile.txt").write_text(
                profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=60
                )
            )
        # Open labels after all measured inference and trace collection.
        labels = torch.load(
            args.workload / "validation-labels.pt", weights_only=False
        )
        stats["quality"] = score(pred.cpu(), labels, task)
        if query_executor is not None:
            query_executor.close()
        if args.mode == "ensemble":
            model.close()
        write_json(args.output / "result.json", stats)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows_per_s": stats["rows_per_s"],
                    "quality": stats["quality"],
                }
            ),
            flush=True,
        )
    finally:
        monitor.terminate()
        monitor.wait(timeout=5)
        telemetry.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--task", default="user-churn")
    prep.add_argument("--entity", default="customer_id")
    prep.add_argument("--entity-table", default="customer")
    prep.add_argument("--time", default="timestamp")
    prep.add_argument("--target", default="churn")
    prep.add_argument(
        "--problem",
        choices=["classification", "regression"],
        default="classification",
    )
    prep.add_argument("--context", type=int, default=1024)
    prep.add_argument("--queries", type=int, default=2000)
    prep.add_argument("--batch-size", type=int, default=250)
    prep.add_argument("--neighbors", type=int, nargs=2, default=[16, 16])
    bench = sub.add_parser("run")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument(
        "--mode", choices=["native", "ensemble", "data"], default="native"
    )
    bench.add_argument("--gpus", type=int, default=1)
    bench.add_argument("--estimators", type=int, default=4)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--warmups", type=int, default=1)
    bench.add_argument("--threads", type=int, default=8)
    bench.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    bench.add_argument("--profile", action="store_true")
    for command in [prep, bench]:
        command.add_argument("--seed", type=int, default=1729)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else run)(args)
    except Exception:
        if args.output.is_dir():
            write_json(
                args.output / "failure.json",
                {
                    "args": vars(args),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
