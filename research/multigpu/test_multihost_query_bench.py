# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU protocol tests; fake workers do not establish GPU scaling."""
# ruff: noqa: D101, D102, D103, TID253

import argparse
import base64
import hashlib
import json
from pathlib import Path
from queue import Queue

import numpy as np
import pytest
from research.multigpu import multihost_query_bench as bench


class FakePeer:
    def __init__(self, config, worker, args):
        self.values = np.load(Path(config["data"]) / "x_val.npy")
        self.pending = Queue()
        self.pending.put({"event": "READY", "context_hashes": {"x": "same"}})
        self.batches = []

    def read(self):
        return self.pending.get(timeout=5)

    def send(self, command):
        if command["op"] == "PREPARE":
            self.batches = command["batches"]
            self.pending.put(
                {
                    "event": "PREPARED",
                    "query_hashes": {
                        str(item["batch_index"]): hashlib.sha256(
                            self.values[item["start"] : item["stop"]].tobytes()
                        ).hexdigest()
                        for item in self.batches
                    },
                }
            )
        elif command["op"] == "MEMORY":
            self.pending.put({"event": "MEMORY", "memory": {}})
        elif command["op"] == "RUN":
            results = []
            for item in self.batches:
                y = np.asarray(item["row_ids"]) % 2
                p = np.stack(
                    [np.where(y == 0, 0.9, 0.1), np.where(y == 1, 0.9, 0.1)],
                    axis=-1,
                ).astype(np.float32)
                results.append(
                    {
                        "batch_index": item["batch_index"],
                        "row_ids": item["row_ids"],
                        "columns": [0, 1],
                        "dtype": str(p.dtype),
                        "shape": p.shape,
                        "data": base64.b64encode(p.tobytes()).decode("ascii"),
                    }
                )
            self.pending.put(
                {"event": "DONE", "results": results, "worker_wall_s": 0.01}
            )

    def close(self):
        pass


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_barrier_transport_and_ordered_gather(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(bench, "Peer", FakePeer)
    np.save(tmp_path / "val_ids.npy", np.arange(9))
    np.save(
        tmp_path / "x_val.npy", np.arange(18, dtype=np.float32).reshape(9, 2)
    )
    np.save(tmp_path / "y_val.npy", np.arange(9) % 2)
    cluster = tmp_path / "cluster.json"
    cluster.write_text(
        json.dumps(
            {
                "data": str(tmp_path),
                "workers": [
                    {"name": f"worker{i}", "device": i} for i in range(4)
                ],
            }
        )
    )
    args = argparse.Namespace(
        output=tmp_path / "result",
        cluster=cluster,
        queries=9,
        batch_size=4,
        workers=workers,
        worker_indices=None,
        weights=None,
        source_commit="test",
        warmups=1,
        repeats=2,
    )
    bench.run(args)
    result = json.loads((args.output / "result.json").read_text())
    assert result["quality"]["accuracy"] == 1
    assert result["repeat_max_abs_difference"] == [0, 0]
    np.testing.assert_array_equal(
        np.load(args.output / "query_ids.npy"), np.arange(9)
    )
    assert np.load(args.output / "predictions.npy").shape == (9, 2)
