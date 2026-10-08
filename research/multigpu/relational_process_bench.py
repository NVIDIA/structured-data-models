# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark spawned KumoRelational replicas on fixed sampled batches."""
# ruff: noqa: T201, D103

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from research.multigpu.process_factories import RelationalProcessFactory
from research.multigpu.query_parallel import ProcessQueryParallel, QueryBatch
from research.multigpu.relational_bench import score, write_json

from sdm import Stype


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    workload = json.loads((args.workload / "workload.json").read_text())
    graphs = torch.load(
        args.workload / "graphs.pt", weights_only=False, map_location="cpu"
    )
    batches = []
    offset = 0
    for sample in graphs["queries"]:
        rows = len(sample.task_table)
        batches.append(
            QueryBatch(
                tuple(range(offset, offset + rows)),
                sample.task_table,
                sample.related_tables,
            )
        )
        offset += rows
    if offset != workload["queries"]:
        raise ValueError("Prepared query count differs from workload manifest")
    devices = [f"cuda:{index}" for index in range(args.gpus)]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[
        args.dtype
    ]
    factory = RelationalProcessFactory(
        graphs=args.workload / "graphs.pt",
        target=workload["target"],
        task=workload["problem"],
        estimators=args.estimators,
        seed=args.seed,
        precision=args.dtype,
        num_hops=2,
        estimator_batch_size=args.estimator_batch_size,
    )
    stats: dict[str, Any] = {
        "args": vars(args),
        "workload": workload,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mode": "process-data",
        "commit": args.source_commit,
        "runner_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "batch_times_kind": "worker_service_time_excluding_queue_and_ipc",
        "predict_timing_boundary": (
            "parent_wall_including_input_ipc_h2d_predict_d2h_output_ipc"
        ),
        "recipe_backend": "cuda",
        "sampling": "fixed_prepared_batches",
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
        start = time.perf_counter()
        with ProcessQueryParallel(
            factory,
            devices,
            dtype=dtype,
            num_threads=args.threads,
        ) as executor:
            executor.ready()
            stats["spawn_load_fit_s"] = time.perf_counter() - start
            stats["memory_after_fit"] = executor.memory()
            start = time.perf_counter()
            for _ in range(args.warmups):
                executor.predict(batches)
            stats["warmup_s"] = time.perf_counter() - start
            executor.memory(reset_peak=True)
            elapsed, gathered_elapsed, batch_times, arrays = [], [], [], []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = executor.predict(batches)
                elapsed.append(time.perf_counter() - start)
                if [item.row_ids for item in result] != [
                    batch.row_ids for batch in batches
                ]:
                    raise ValueError(
                        "Process output observation order changed"
                    )
                batch_times.append([item.seconds for item in result])
                pred = torch.cat([item.prediction for item in result], dim=0)
                gathered_elapsed.append(time.perf_counter() - start)
                arrays.append(pred.numerical.numpy().copy())
            stats["memory_prediction"] = executor.memory()
        # All child models are gone before validation targets become visible.
        stats["predict_repeats_s"] = elapsed
        stats["gathered_output_repeats_s"] = gathered_elapsed
        stats["gathered_rows_per_s"] = [
            workload["queries"] / seconds for seconds in gathered_elapsed
        ]
        stats["batch_times_s"] = batch_times
        stats["rows_per_s"] = [
            workload["queries"] / seconds for seconds in elapsed
        ]
        stats["repeat_max_abs_difference"] = [
            float(np.abs(array - arrays[0]).max()) for array in arrays
        ]
        stats["prediction_sha256"] = hashlib.sha256(
            arrays[0].tobytes()
        ).hexdigest()
        stats["prediction_repeat_sha256"] = [
            hashlib.sha256(array.tobytes()).hexdigest() for array in arrays
        ]
        for repeat, array in enumerate(arrays):
            np.save(args.output / f"predictions-repeat-{repeat}.npy", array)
        stats["prediction_columns"] = list(pred.columns[Stype.numerical])
        np.save(args.output / "predictions.npy", arrays[0])
        torch.save(pred, args.output / "predictions.pt")
        labels = torch.load(
            args.workload / "validation-labels.pt",
            weights_only=False,
            map_location="cpu",
        )
        stats["quality"] = score(pred, labels, workload["problem"])
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
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--estimator-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    args = parser.parse_args()
    try:
        run(args)
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
