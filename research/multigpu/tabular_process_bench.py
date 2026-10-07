# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure process DP with fixed query batches and CPU-complete output."""
# ruff: noqa: D103, T201

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from research.multigpu.process_factories import TabularProcessFactory
from research.multigpu.query_parallel import ProcessQueryParallel, QueryBatch
from research.multigpu.relational_bench import score, write_json

from sdm import CategoricalTensor, Stype, TableTensor


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    values = np.load(args.data / "x_val.npy", mmap_mode="r")[
        : args.queries
    ].copy()
    ids = np.load(args.data / "val_ids.npy", mmap_mode="r")[
        : args.queries
    ].copy()
    if len(values) != args.queries or len(ids) != args.queries:
        raise ValueError(
            "Requested queries exceed the prepared validation split"
        )
    batches = [
        QueryBatch(
            tuple(
                int(value) for value in ids[start : start + args.batch_size]
            ),
            TableTensor.from_tensor(
                torch.from_numpy(
                    values[start : start + args.batch_size]
                ).float()
            ),
        )
        for start in range(0, args.queries, args.batch_size)
    ]
    factory = TabularProcessFactory(
        data=args.data,
        context=args.context,
        task=args.task,
        size=args.size,
        estimators=args.estimators,
        seed=args.seed,
        precision=args.precision,
        estimator_batch_size=args.estimator_batch_size,
    )
    dtype = torch.bfloat16 if args.precision == "bfloat16" else None
    report = {
        "config": vars(args),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_revision": args.source_commit,
            "runner_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "mode": "process-data",
        "input_residency": "prepared_CPU_batches",
        "timing": "parent_wall_including_input_IPC_H2D_predict_D2H_output_IPC",
        "batch_times_kind": "worker_service_time_excluding_queue_and_ipc",
        "query_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "query_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
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
            [f"cuda:{i}" for i in range(args.gpus)],
            dtype=dtype,
            num_threads=args.threads,
        ) as executor:
            executor.ready()
            report["spawn_load_fit_s"] = time.perf_counter() - start
            report["memory_after_fit"] = executor.memory()
            start = time.perf_counter()
            for _ in range(args.warmups):
                executor.predict(batches)
            report["warmup_s"] = time.perf_counter() - start
            executor.memory(reset_peak=True)
            elapsed, gathered_elapsed, durations, arrays = [], [], [], []
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
                durations.append([item.seconds for item in result])
                prediction = torch.cat(
                    [item.prediction for item in result], dim=0
                )
                gathered_elapsed.append(time.perf_counter() - start)
                arrays.append(prediction.numerical.float().numpy().copy())
            report["memory_after_prediction"] = executor.memory()
        report["predict_repeats_s"] = elapsed
        report["gathered_output_repeats_s"] = gathered_elapsed
        report["gathered_rows_per_s"] = [
            args.queries / seconds for seconds in gathered_elapsed
        ]
        report["rows_per_s"] = [args.queries / seconds for seconds in elapsed]
        report["batch_times_s"] = durations
        report["repeat_max_abs_difference"] = [
            float(np.abs(array - arrays[0]).max()) for array in arrays
        ]
        report["prediction_columns"] = list(
            prediction.columns[Stype.numerical]
        )
        report["prediction_sha256"] = hashlib.sha256(
            arrays[0].tobytes()
        ).hexdigest()
        np.save(args.output / "predictions.npy", arrays[0])
        np.save(args.output / "query_ids.npy", ids)
        # Worker teardown precedes opening validation targets.
        targets = np.load(args.data / "y_val.npy", mmap_mode="r")[
            : args.queries
        ].copy()
        target_values = torch.from_numpy(targets).reshape(-1, 1)
        labels = (
            TableTensor(
                columns={Stype.categorical: ("target",)},
                categorical=CategoricalTensor.from_tensor(
                    target_values.long()
                ),
            )
            if args.task == "classification"
            else TableTensor.from_tensor(target_values.float())
        )
        report["quality"] = score(prediction, labels, args.task)
        np.save(args.output / "targets.npy", targets)
        write_json(args.output / "result.json", report)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows_per_s": report["rows_per_s"],
                    "quality": report["quality"],
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--task", choices=["classification", "regression"], required=True
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large"], default="large"
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--estimator-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--precision", choices=["bfloat16", "float32"], default="bfloat16"
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
