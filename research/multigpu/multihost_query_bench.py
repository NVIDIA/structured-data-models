# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate fixed query shards over persistent local/SSH GPU workers."""
# ruff: noqa: D101, D102, D103, T201

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from research.multigpu.query_shards import (
    gather_batches,
    plan_batches,
    weighted_plan_batches,
)
from research.multigpu.relational_bench import score, write_json

from sdm import CategoricalTensor, Stype, TableTensor


class Peer:
    def __init__(
        self, config: dict, worker: dict, args: argparse.Namespace
    ) -> None:
        self.worker = worker
        env = [
            "env",
            f"CUDA_VISIBLE_DEVICES={worker['device']}",
            f"PYTHONPATH={config['source']}",
            f"HF_HUB_CACHE={config['hf_cache']}",
            "HF_HUB_OFFLINE=1",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
        ]
        command = [
            *env,
            config["python"],
            "-u",
            "-m",
            "research.multigpu.ssh_query_worker",
            "--data",
            config["data"],
            "--context",
            str(args.context),
            "--estimators",
            str(args.estimators),
            "--estimator-batch-size",
            str(args.estimator_batch_size),
            "--seed",
            str(args.seed),
            "--size",
            args.size,
            "--precision",
            args.precision,
            "--threads",
            str(args.threads),
        ]
        if worker.get("host"):
            command = [
                "ssh",
                "-i",
                config["identity"],
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                f"ubuntu@{worker['host']}",
                shlex.join(command),
            ]
        self.log = (args.output / f"{worker['name']}.stderr.log").open("w")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )

    def send(self, command: dict) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()

    def read(self) -> dict:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("SDM_RPC "):
                return json.loads(line[len("SDM_RPC ") :])
        raise RuntimeError(
            f"Worker {self.worker['name']} exited; inspect stderr log"
        )

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.send({"op": "STOP"})
                self.process.wait(timeout=20)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=20)
            self.log.close()


def exchange(
    peers: list[Peer], commands: list[dict], pool: ThreadPoolExecutor
) -> list[dict]:
    futures = [pool.submit(peer.read) for peer in peers]
    for peer, command in zip(peers, commands, strict=True):
        peer.send(command)
    return [future.result(timeout=900) for future in futures]


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.cluster.read_text())
    workers = (
        [config["workers"][index] for index in args.worker_indices]
        if args.worker_indices is not None
        else config["workers"][: args.workers]
    )
    if args.worker_indices is None and len(workers) != args.workers:
        raise ValueError("Cluster does not contain the requested workers")
    worker_count = len(workers)
    if (
        len({(worker.get("host"), worker["device"]) for worker in workers})
        != worker_count
    ):
        raise ValueError("Each worker must own a distinct host/GPU pair")
    data = Path(config["data"])
    ids = np.load(data / "val_ids.npy", mmap_mode="r")[: args.queries].copy()
    values = np.load(data / "x_val.npy", mmap_mode="r")[: args.queries]
    if len(ids) != args.queries or len(values) != args.queries:
        raise ValueError("Insufficient prepared validation observations")
    rows = [
        ids[start : start + args.batch_size].tolist()
        for start in range(0, args.queries, args.batch_size)
    ]
    if args.weights is not None and len(args.weights) != worker_count:
        raise ValueError("Provide one measured capacity weight per worker")
    assignments = (
        weighted_plan_batches(rows, args.weights)
        if args.weights is not None
        else plan_batches(rows, worker_count)
    )
    expected_hashes = {
        str(index): hashlib.sha256(
            values[
                index * args.batch_size : (index + 1) * args.batch_size
            ].tobytes()
        ).hexdigest()
        for index in range(len(rows))
    }
    peers: list[Peer] = []
    report: dict[str, Any] = {
        "args": vars(args),
        "cluster": config,
        "workers": workers,
        "source_commit": args.source_commit,
        "assignments": assignments,
        "query_ids": ids.tolist(),
        "timing": (
            "single_coordinator_monotonic_wall_dispatch_network_decode_ordered_gather"
        ),
        "input_residency": "preloaded_CPU_fixed_batches_per_worker",
        "clock_assumption": "none; worker clocks never summed or compared",
    }
    try:
        start = time.perf_counter()
        for worker in workers:
            peers.append(Peer(config, worker, args))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            ready = [
                future.result(timeout=900)
                for future in [pool.submit(peer.read) for peer in peers]
            ]
            if any(item["event"] != "READY" for item in ready):
                raise ValueError("Not every worker completed setup")
            uuids = [item.get("gpu_uuid", "unavailable") for item in ready]
            if "unavailable" not in uuids and len(set(uuids)) != worker_count:
                raise ValueError("Workers resolved to duplicate physical GPUs")
            if any(
                item["context_hashes"] != ready[0]["context_hashes"]
                for item in ready
            ):
                raise ValueError("Workers fitted different context inputs")
            report["ready"] = ready
            report["startup_load_fit_s"] = time.perf_counter() - start
            prepared = exchange(
                peers,
                [
                    {
                        "op": "PREPARE",
                        "batches": [
                            {
                                "batch_index": index,
                                "row_ids": rows[index],
                                "start": index * args.batch_size,
                                "stop": min(
                                    (index + 1) * args.batch_size, args.queries
                                ),
                            }
                            for index in assignment
                        ],
                    }
                    for assignment in assignments
                ],
                pool,
            )
            observed_hashes = {
                key: value
                for item in prepared
                for key, value in item["query_hashes"].items()
            }
            if observed_hashes != expected_hashes:
                raise ValueError("Worker query inputs differ from coordinator")
            report["query_hashes"] = observed_hashes
            elapsed, worker_times, arrays = [], [], []
            for repeat in range(args.warmups + args.repeats):
                if repeat == args.warmups:
                    exchange(
                        peers,
                        [{"op": "MEMORY", "reset_peak": True}] * len(peers),
                        pool,
                    )
                start = time.perf_counter()
                messages = exchange(peers, [{"op": "RUN"}] * len(peers), pool)
                results = []
                for message in messages:
                    if message["event"] != "DONE":
                        raise ValueError("Unexpected worker response")
                    for result in message["results"]:
                        result["predictions"] = np.frombuffer(
                            base64.b64decode(result.pop("data")),
                            dtype=result.pop("dtype"),
                        ).reshape(result.pop("shape"))
                        results.append(result)
                columns = results[0]["columns"]
                prediction = gather_batches(results, rows, columns)
                seconds = time.perf_counter() - start
                if repeat >= args.warmups:
                    elapsed.append(seconds)
                    worker_times.append(
                        [message["worker_wall_s"] for message in messages]
                    )
                    arrays.append(prediction.copy())
            report["memory_prediction"] = exchange(
                peers, [{"op": "MEMORY"}] * len(peers), pool
            )
        for peer in peers:
            peer.close()
        peers = []
        report["predict_repeats_s"] = elapsed
        report["rows_per_s"] = [args.queries / value for value in elapsed]
        report["worker_wall_s"] = worker_times
        report["prediction_columns"] = columns
        report["repeat_max_abs_difference"] = [
            float(np.abs(array - arrays[0]).max()) for array in arrays
        ]
        report["prediction_sha256"] = hashlib.sha256(
            arrays[0].tobytes()
        ).hexdigest()
        np.save(args.output / "predictions.npy", arrays[0])
        np.save(args.output / "query_ids.npy", ids)
        targets = np.load(data / "y_val.npy", mmap_mode="r")[
            : args.queries
        ].copy()
        labels = TableTensor(
            columns={Stype.categorical: ("target",)},
            categorical=CategoricalTensor.from_tensor(
                torch.from_numpy(targets).long().reshape(-1, 1)
            ),
        )
        predictions = TableTensor(
            columns={Stype.numerical: columns},
            numerical=torch.from_numpy(arrays[0]),
        )
        report["quality"] = score(predictions, labels, "classification")
        write_json(args.output / "result.json", report)
        print(
            json.dumps(
                {
                    "rows_per_s": report["rows_per_s"],
                    "quality": report["quality"],
                }
            ),
            flush=True,
        )
    finally:
        for peer in peers:
            peer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-indices", type=int, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--queries", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--size", default="large")
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--estimator-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--precision", choices=["bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        if args.output.is_dir():
            write_json(
                args.output / "failure.json",
                {"args": vars(args), "traceback": traceback.format_exc()},
            )
        raise


if __name__ == "__main__":
    main()
