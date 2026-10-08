# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose existing graph replay with persistent query workers."""
# ruff: noqa: D101, D102, D103, TID253

import os
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from research.multigpu.process_factories import TabularProcessFactory
from research.multigpu.query_parallel import ProcessQueryParallel, QueryBatch

from sdm import TableTensor


@dataclass
class ClosingPredictor:
    marker: Path

    def predict(self, x, related_tables=None):
        return x.clone()

    def close(self):
        self.marker.write_text("closed")


@dataclass
class ClosingFactory:
    directory: Path

    def __call__(self, worker, device):
        return ClosingPredictor(self.directory / f"closed-{worker}")


class KilledPredictor:
    def predict(self, x, related_tables=None):
        os._exit(7)


def killed_factory(worker, device):
    return KilledPredictor()


def test_killed_worker_cleanup_does_not_replace_inference_failure():
    with ProcessQueryParallel(killed_factory, ["cpu"]) as executor:
        executor.ready()
        with pytest.raises(BrokenProcessPool):
            executor.predict(
                [QueryBatch((0,), TableTensor.from_tensor(torch.ones(1, 1)))]
            )
    # Cleanup of the failed pool was successful and remains idempotent.
    executor.close()


def test_process_shutdown_closes_nested_predictors(tmp_path):
    executor = ProcessQueryParallel(ClosingFactory(tmp_path), ["cpu", "cpu"])
    executor.ready()
    executor.close()
    executor.close()  # Idempotent; no new process can be started on cleanup.
    assert [(tmp_path / f"closed-{i}").read_text() for i in range(2)] == [
        "closed",
        "closed",
    ]


def test_graph_backend_rejects_cpu_before_loading_weights(tmp_path):
    factory = TabularProcessFactory(tmp_path, context=32, backend="graph")
    with pytest.raises(ValueError, match="requires CUDA"):
        factory(0, torch.device("cpu"))


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [256, 1024])
def test_cuda_process_graph_fixed_and_irregular_shapes(
    tmp_path, workers, batch_size
):
    if torch.cuda.device_count() < workers:
        pytest.skip(f"requires {workers} CUDA devices")
    generator = np.random.default_rng(1729)
    np.save(
        tmp_path / "x_train.npy",
        generator.normal(size=(32, 4)).astype("float32"),
    )
    np.save(tmp_path / "y_train.npy", np.arange(32) % 3)
    query_count = 2 * workers * batch_size + 3
    values = generator.normal(size=(query_count, 4)).astype("float32")
    batches = [
        QueryBatch(
            tuple(range(start, min(start + batch_size, query_count))),
            TableTensor.from_tensor(
                torch.from_numpy(values[start : start + batch_size])
            ),
        )
        for start in range(0, query_count, batch_size)
    ]
    devices = [f"cuda:{index}" for index in range(workers)]
    options = {
        "data": tmp_path,
        "context": 32,
        "size": "small",
        "estimators": 2,
        "pretrained": False,
        "seed": 1729,
        "precision": "bfloat16",
    }
    with ProcessQueryParallel(
        TabularProcessFactory(**options), ["cuda:0"]
    ) as oracle:
        oracle.ready()
        expected = oracle.predict(batches)
    with ProcessQueryParallel(
        TabularProcessFactory(**options, backend="graph"), devices
    ) as executor:
        executor.ready()
        assert all(item["graph_count"] == 0 for item in executor.memory())
        warm = executor.predict(batches)
        counts = [item["graph_count"] for item in executor.memory()]
        assert all(count >= 2 for count in counts)
        # Only worker zero receives and captures the final partial batch.
        assert counts[0] == 4
        preserved = [item.prediction.numerical.clone() for item in warm]
        for _ in range(3):
            observed = executor.predict(batches)
            assert [
                item["graph_count"] for item in executor.memory()
            ] == counts
            assert [item.row_ids for item in observed] == [
                item.row_ids for item in batches
            ]
            for candidate, reference in zip(observed, expected, strict=True):
                torch.testing.assert_close(
                    candidate.prediction.numerical,
                    reference.prediction.numerical,
                    atol=1e-2,
                    rtol=5e-2,
                )
        for original, saved in zip(warm, preserved, strict=True):
            torch.testing.assert_close(
                original.prediction.numerical, saved, atol=0, rtol=0
            )
