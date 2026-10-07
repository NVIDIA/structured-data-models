"""Pretrained Kumo context-parallel benchmark on fixed validation workloads.

Use a fresh torchrun invocation for each native, single-rank LSE, or CP arm.
All ranks receive identical context/query rows; throughput counts rows once.
"""

# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.models import KumoRelational, KumoTabular
from sdm.nn.context_parallel import context_parallel


def icl_bytes(value: object) -> int:
    """Count ICL cache tensors separately from replicated encoder state."""
    if isinstance(value, Mapping):
        return sum(
            sum(t.numel() * t.element_size() for t in item)
            if isinstance(item, KVCacheEntry) and "icl_block" in str(key)
            else icl_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(icl_bytes(item) for item in value)
    return 0


def cache_storage(cache: Cache) -> dict[str, int]:
    """Count unique cache backing storage by actual device residency."""
    seen, result = set(), {}
    for tensor in cache._tensors():
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            result[key[0]] = result.get(key[0], 0) + storage.nbytes()
    return result


def main() -> None:
    """Run one arm, retaining rank timing, predictions, IDs, and quality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--family", choices=["tabular", "relational"], required=True
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="small"
    )
    parser.add_argument(
        "--mode", choices=["native", "lse", "context"], default="context"
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--reduction",
        choices=["all_reduce", "all_gather"],
        default="all_reduce",
    )
    parser.add_argument(
        "--cache-residency", choices=["default", "resident"], default="default"
    )
    parser.add_argument(
        "--kernel",
        choices=["efficient", "flash", "efficient_fp32"],
        default="efficient",
    )
    parser.add_argument(
        "--source-commit", help="Revision for archive deployments"
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if args.mode != "context" and world != 1:
        raise ValueError("Native and LSE controls require exactly one rank")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    try:
        if args.family == "tabular":
            x = torch.from_numpy(
                np.load(args.data / "x_train.npy", mmap_mode="r")[
                    : args.context
                ].copy()
            ).float()
            y = torch.from_numpy(
                np.load(args.data / "y_train.npy", mmap_mode="r")[
                    : args.context
                ].copy()
            ).reshape(-1, 1)
            query = torch.from_numpy(
                np.load(args.data / "x_val.npy", mmap_mode="r")[
                    : args.queries
                ].copy()
            ).float()
            if len(x) != args.context or len(query) != args.queries:
                raise ValueError(
                    "Requested rows exceed the fixed TRAIN/VAL split"
                )
            input_identity = {
                "x_context_sha256": hashlib.sha256(
                    x.numpy().tobytes()
                ).hexdigest(),
                "y_context_sha256": hashlib.sha256(
                    y.numpy().tobytes()
                ).hexdigest(),
                "x_query_sha256": hashlib.sha256(
                    query.numpy().tobytes()
                ).hexdigest(),
                "context_rows": len(x),
                "query_rows": len(query),
            }
            x = TableTensor.from_tensor(x.to(device))
            y = (
                TableTensor(
                    columns={Stype.categorical: ["target"]},
                    categorical=CategoricalTensor.from_tensor(
                        y.long().to(device)
                    ),
                )
                if args.task == "classification"
                else TableTensor.from_tensor(y.float().to(device))
            )
            batches = [
                (batch,)
                for batch in TableTensor.from_tensor(query.to(device)).split(
                    args.batch_size
                )
            ]
            fit_args = (x, y)
            extra = {}
            rows = len(query)
            query_ids = np.load(args.data / "val_ids.npy")[:rows]
        else:
            manifest = json.loads((args.data / "workload.json").read_text())
            graphs = torch.load(args.data / "graphs.pt", weights_only=False)
            context = graphs["context"].to(device)
            fit_args = (
                context.task_table.drop_columns(manifest["target"]),
                context.task_table[manifest["target"]],
                context.related_tables,
            )
            batches = [tuple(batch.to(device)) for batch in graphs["queries"]]
            extra = {"num_hops": 2}
            rows = sum(len(batch[0]) for batch in batches)
            query_ids = np.arange(rows)
            args.task = manifest["problem"]
            input_identity = {
                "workload": manifest,
                "workload_sha256": hashlib.sha256(
                    (args.data / "workload.json").read_bytes()
                ).hexdigest(),
            }
        input_identity["actual_batch_rows"] = [
            len(batch[0]) for batch in batches
        ]
        input_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = (
            KumoTabular(task=args.task, size=args.size, device=device)
            if args.family == "tabular"
            else KumoRelational(task=args.task, device=device)
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        scope = (
            nullcontext()
            if args.mode == "native"
            else context_parallel(
                dist.group.WORLD, kernel=args.kernel, reduction=args.reduction
            )
        )
        dtype = getattr(torch, args.precision)
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ),
            scope,
        ):
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            torch.cuda.nvtx.range_push("context_fit")
            model.fit(
                *fit_args,
                num_estimators=args.estimators,
                estimator_batch_size=1,
                generator=torch.Generator(device).manual_seed(args.seed),
                **extra,
            )
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            fit_seconds = time.perf_counter() - started
            fit_peak = torch.cuda.max_memory_allocated()
            residency_start = time.perf_counter()
            if args.cache_residency == "resident":
                model._cache = model._cache.to(device)
            torch.cuda.synchronize()
            residency_seconds = time.perf_counter() - residency_start
            residency_peak = torch.cuda.max_memory_allocated()
            cache = model._cache
            cache_bytes = cache.size() if isinstance(cache, Cache) else None
            cache_icl_bytes = icl_bytes(cache)
            cache_by_device = (
                cache_storage(cache) if isinstance(cache, Cache) else {}
            )
            fit_reserved = torch.cuda.max_memory_reserved()
            model.predict(*batches[0])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            pass_times, batch_times, repeat_hashes = [], [], []
            for repeat in range(args.repeats):
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                predictions, durations = [], []
                for batch in batches:
                    batch_start = time.perf_counter()
                    torch.cuda.nvtx.range_push("context_predict_batch")
                    pred = model.predict(*batch)
                    predictions.append(pred.cpu())
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                    durations.append(time.perf_counter() - batch_start)
                pass_times.append(time.perf_counter() - started)
                batch_times.append(durations)
                combined = torch.cat(predictions, dim=0)
                repeat_hashes.append(
                    hashlib.sha256(
                        combined.numerical.float().numpy().tobytes()
                    ).hexdigest()
                )
                if rank == 0:
                    np.save(
                        args.output / f"predictions-{repeat}.npy",
                        combined.numerical.float().numpy(),
                    )
            prediction_peak = torch.cuda.max_memory_allocated()
            prediction_reserved = torch.cuda.max_memory_reserved()
            prediction_current = torch.cuda.memory_allocated()
            prediction_current_reserved = torch.cuda.memory_reserved()
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    model.predict(*batches[0])
                    torch.cuda.synchronize()
                profile.export_chrome_trace(
                    str(args.output / f"profile-rank{rank}.json")
                )
        record = {
            "rank": rank,
            "load_seconds": load_seconds,
            "fit_seconds": fit_seconds,
            "fit_peak_bytes": fit_peak,
            "residency_seconds": residency_seconds,
            "fit_and_residency_peak_bytes": residency_peak,
            "cache_bytes": cache_bytes,
            "input_allocated_bytes": input_allocated,
            "cache_storage_bytes_by_device": cache_by_device,
            "fit_peak_reserved_bytes": fit_reserved,
            "icl_cache_bytes": cache_icl_bytes,
            "prediction_peak_bytes": prediction_peak,
            "prediction_peak_reserved_bytes": prediction_reserved,
            "current_allocated_bytes": prediction_current,
            "current_reserved_bytes": prediction_current_reserved,
            "repeat_prediction_sha256": repeat_hashes,
            "pass_seconds": pass_times,
            "batch_seconds": batch_times,
            "prediction_sha256": hashlib.sha256(
                combined.numerical.float().numpy().tobytes()
            ).hexdigest(),
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        gathered = [None] * world
        dist.all_gather_object(gathered, record)
        if len({record["prediction_sha256"] for record in gathered}) != 1:
            raise RuntimeError(
                "Replicated ranks produced different predictions"
            )
        if rank == 0:
            # Validation targets are read only after predictions complete.
            if args.family == "tabular":
                from research.multigpu.tabular_bench import (
                    quality,
                )

                target = np.load(args.data / "y_val.npy")[:rows]
                columns = list(combined.columns[Stype.numerical])
                quality_result = quality(
                    args.task,
                    target,
                    combined.numerical.float().numpy(),
                    columns,
                )
            else:
                from research.multigpu.relational_bench import (
                    score,
                )

                target = torch.load(
                    args.data / "validation-labels.pt", weights_only=False
                )
                quality_result = score(combined, target, args.task)
                columns = list(combined.columns[Stype.numerical])
            np.save(args.output / "query_ids.npy", query_ids)
            slowest = [
                max(record["pass_seconds"][i] for record in gathered)
                for i in range(args.repeats)
            ]
            result = {
                "status": "complete",
                "input_identity": input_identity,
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "world": world,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "revision": args.source_commit
                or subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    text=True,
                    capture_output=True,
                    check=False,
                ).stdout.strip()
                or "unknown",
                "ranks": gathered,
                "slowest_rank_pass_seconds": slowest,
                "unique_rows_per_second": [
                    rows / seconds for seconds in slowest
                ],
                "median_unique_rows_per_second": rows
                / statistics.median(slowest),
                "quality": quality_result,
                "columns": columns,
                "query_ids_sha256": hashlib.sha256(
                    query_ids.tobytes()
                ).hexdigest(),
            }
            (args.output / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    except Exception as error:
        (args.output / f"error-rank{rank}.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "rank": rank,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
