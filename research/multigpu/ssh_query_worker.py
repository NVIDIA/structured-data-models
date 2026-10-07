# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent inference worker controlled through SSH stdin/stdout JSON."""
# ruff: noqa: D103, T201

import argparse
import base64
import hashlib
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from research.multigpu.process_factories import TabularProcessFactory

from sdm import Stype, TableTensor

PREFIX = "SDM_RPC "


def emit(value: dict) -> None:
    print(PREFIX + json.dumps(value, default=str), flush=True)


def memory() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--size", default="large")
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--estimator-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--precision", default="bfloat16")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(0)
    factory = TabularProcessFactory(
        data=args.data,
        context=args.context,
        size=args.size,
        estimators=args.estimators,
        estimator_batch_size=args.estimator_batch_size,
        seed=args.seed,
        precision=args.precision,
    )
    start = time.perf_counter()
    model = factory(0, torch.device("cuda:0"))
    torch.cuda.synchronize()
    context_hashes = {
        name: hashlib.sha256(
            np.load(args.data / f"{name}.npy", mmap_mode="r")[
                : args.context
            ].tobytes()
        ).hexdigest()
        for name in ["x_train", "y_train"]
    }
    emit(
        {
            "event": "READY",
            "load_fit_s": time.perf_counter() - start,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": str(
                getattr(
                    torch.cuda.get_device_properties(0), "uuid", "unavailable"
                )
            ),
            "gpu_total_bytes": torch.cuda.get_device_properties(
                0
            ).total_memory,
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "threads": torch.get_num_threads(),
            "context_hashes": context_hashes,
            "memory": memory(),
        }
    )
    batches = []
    dtype = torch.bfloat16 if args.precision == "bfloat16" else None
    for line in sys.stdin:
        command = json.loads(line)
        if command["op"] == "STOP":
            emit({"event": "STOPPED"})
            return
        if command["op"] == "PREPARE":
            values = np.load(args.data / "x_val.npy", mmap_mode="r")
            ids = np.load(args.data / "val_ids.npy", mmap_mode="r")
            batches = []
            hashes = {}
            for item in command["batches"]:
                start, stop = item["start"], item["stop"]
                if ids[start:stop].tolist() != item["row_ids"]:
                    raise ValueError("Query observation IDs differ on worker")
                array = values[start:stop].copy()
                hashes[str(item["batch_index"])] = hashlib.sha256(
                    array.tobytes()
                ).hexdigest()
                batches.append(
                    (
                        item,
                        TableTensor.from_tensor(
                            torch.from_numpy(array).float()
                        ),
                    )
                )
            emit({"event": "PREPARED", "query_hashes": hashes})
        elif command["op"] == "MEMORY":
            emit({"event": "MEMORY", "memory": memory()})
            if command.get("reset_peak"):
                torch.cuda.reset_peak_memory_stats()
        elif command["op"] == "RUN":
            outputs = []
            start = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast(
                    "cuda",
                    dtype=dtype,
                    enabled=dtype is not None,
                ),
            ):
                for item, x in batches:
                    batch_start = time.perf_counter()
                    prediction = model.predict(x.to("cuda:0")).cpu()
                    seconds = time.perf_counter() - batch_start
                    array = prediction.numerical.float().numpy()
                    outputs.append(
                        {
                            "batch_index": item["batch_index"],
                            "row_ids": item["row_ids"],
                            "columns": list(
                                prediction.columns[Stype.numerical]
                            ),
                            "shape": list(array.shape),
                            "dtype": str(array.dtype),
                            "data": base64.b64encode(array.tobytes()).decode(
                                "ascii"
                            ),
                            "service_s": seconds,
                        }
                    )
            emit(
                {
                    "event": "DONE",
                    "worker_wall_s": time.perf_counter() - start,
                    "results": outputs,
                }
            )
        else:
            raise ValueError(f"Unknown operation {command['op']}")


if __name__ == "__main__":
    main()
