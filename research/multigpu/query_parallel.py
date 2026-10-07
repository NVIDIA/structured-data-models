# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference data parallelism over complete prepared query batches.

Models must already be fitted to the same context/recipe/member plan. Each
worker may itself be an EnsembleParallel executor for DP x EP composition.
"""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from typing import Protocol

import torch

from sdm import RelatedTables, TableTensor


class Predictor(Protocol):
    """The fitted inference API required by both executors."""

    def predict(
        self,
        x: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        """Predict a complete batch using already fitted context."""
        ...


@dataclass(frozen=True)
class QueryBatch:
    """An immutable scheduling unit with its complete relational neighborhood.

    row_ids are external observation IDs, not entity IDs: repeated entities at
    different prediction times must have distinct observation IDs.
    """

    row_ids: tuple[int, ...]
    x: TableTensor
    related_tables: RelatedTables | None = None


@dataclass(frozen=True)
class QueryResult:
    """Ordered observations, CPU predictions, worker and service duration."""

    row_ids: tuple[int, ...]
    prediction: TableTensor
    worker: int
    seconds: float


def _validate_batch(batch: QueryBatch) -> None:
    if len(batch.row_ids) != batch.x.size(-2):
        raise ValueError("Observation IDs must match the query row count")
    if batch.x.device.type != "cpu" or (
        batch.related_tables is not None
        and any(
            table.device.type != "cpu"
            for table in batch.related_tables.tables.values()
        )
    ):
        raise ValueError("Prepared query batches must reside on CPU")


def _collect(futures: Sequence[Future]) -> list[QueryResult]:
    # A failed request may be retried/refitted after this call returns. Drain
    # all sibling work first so the old request cannot touch those caches.
    wait(futures)
    results: list[QueryResult] = [future.result() for future in futures]
    if results and any(
        result.prediction.columns != results[0].prediction.columns
        for result in results[1:]
    ):
        raise ValueError(
            "Replica predictions must have identical output columns"
        )
    return results


def _predict_batch(
    model: Predictor,
    batch: QueryBatch,
    worker: int,
    device: torch.device,
    dtype: torch.dtype | None,
) -> QueryResult:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    start = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype is not None and device.type == "cuda",
        ),
    ):
        x = batch.x.to(device)
        related = (
            None
            if batch.related_tables is None
            else batch.related_tables.to(device)
        )
        out = model.predict(x, related).to("cpu")
    if out.size(-2) != len(batch.row_ids):
        raise ValueError("Prediction must preserve query row count")
    return QueryResult(batch.row_ids, out, worker, time.perf_counter() - start)


class QueryParallel:
    """Run complete query batches on independently fitted replicas.

    Args:
        models: Independent fitted models or ensemble executors. A model must
            not be shared between workers or concurrently used externally.
        devices: Input device for each model; ensemble workers can own more
            than one device internally.
        dtype: Explicit worker autocast dtype, or None to disable autocast.
            Thread-local caller autocast and inference state are not inherited.

    The persistent single-thread queue per worker serializes access to that
    worker's mutable recipe/cache state, including concurrent submit callers.
    Results return on CPU, so completion includes device synchronization and
    output transfer. Already prepared inputs must be CPU resident; their
    storage must remain unchanged until all submitted futures complete.
    """

    def __init__(
        self,
        models: Sequence[Predictor],
        devices: Sequence[torch.device | str],
        *,
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        if not models or len(models) != len(devices):
            raise ValueError("Provide one device for each fitted model")
        if len({id(model) for model in models}) != len(models):
            raise ValueError("Workers must own independent model instances")
        self.models = tuple(models)
        self.devices = tuple(torch.device(device) for device in devices)
        self.dtype = dtype
        # Explicit setup boundary: fit may have used another CUDA stream.
        for device in self.devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        self._pools = [ThreadPoolExecutor(max_workers=1) for _ in models]

    def submit(self, batch: QueryBatch, worker: int) -> Future[QueryResult]:
        """Submit to a named replica, preserving the complete batch graph."""
        _validate_batch(batch)
        return self._pools[worker].submit(self._predict, batch, worker)

    def predict(self, batches: Sequence[QueryBatch]) -> list[QueryResult]:
        """Schedule round robin and return results in original batch order."""
        futures = [
            self.submit(batch, index % len(self.models))
            for index, batch in enumerate(batches)
        ]
        return _collect(futures)

    def _predict(self, batch: QueryBatch, worker: int) -> QueryResult:
        return _predict_batch(
            self.models[worker],
            batch,
            worker,
            self.devices[worker],
            self.dtype,
        )

    def close(self) -> None:
        """Drain pending work and release worker threads."""
        for pool in self._pools:
            pool.shutdown(wait=True)

    def __enter__(self) -> QueryParallel:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


_process_state: (
    tuple[Predictor, int, torch.device, torch.dtype | None] | None
) = None


def _initialize_process(
    factory: Callable[[int, torch.device], Predictor],
    worker: int,
    device: torch.device,
    dtype: torch.dtype | None,
    num_threads: int,
) -> None:
    global _process_state
    torch.set_num_threads(num_threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _process_state = (factory(worker, device), worker, device, dtype)


def _process_predict(batch: QueryBatch | None) -> QueryResult | None:
    assert _process_state is not None
    if batch is None:
        return None
    model, worker, device, dtype = _process_state
    return _predict_batch(model, batch, worker, device, dtype)


class ProcessQueryParallel:
    """Spawn one process per replica, fitting through a picklable factory.

    Args:
        factory: Module-level callable (worker index, input device) -> fitted
            Predictor. It must recreate identical weights/context/recipe/RNG
            state regardless of worker index. No parent CUDA state is forked.
        devices: One input device per replica or hybrid ensemble group.
        dtype: Explicit inference autocast dtype, or None.
        num_threads: Intra-op CPU threads per child to avoid oversubscription.

    Call ready() before warm timing to separate Python startup/checkpoint/fit.
    Parent wall timing includes serialization/IPC; QueryResult.seconds does
    not. Large RelatedTables can make that distinction substantial.
    """

    def __init__(
        self,
        factory: Callable[[int, torch.device], Predictor],
        devices: Sequence[torch.device | str],
        *,
        dtype: torch.dtype | None = torch.bfloat16,
        num_threads: int = 1,
    ) -> None:
        if not devices:
            raise ValueError("Provide at least one worker device")
        self._pools = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process,
                initargs=(
                    factory,
                    worker,
                    torch.device(device),
                    dtype,
                    num_threads,
                ),
            )
            for worker, device in enumerate(devices)
        ]

    def ready(self) -> None:
        """Wait for every process to finish model construction and fitting."""
        futures = [pool.submit(_process_predict, None) for pool in self._pools]
        for future in futures:
            future.result()

    def predict(self, batches: Sequence[QueryBatch]) -> list[QueryResult]:
        """Predict complete batches and return results in their input order."""
        for batch in batches:
            _validate_batch(batch)
        futures = [
            self._pools[index % len(self._pools)].submit(
                _process_predict, batch
            )
            for index, batch in enumerate(batches)
        ]
        return _collect(futures)

    def close(self) -> None:
        """Drain pending work and stop each worker process."""
        for pool in self._pools:
            pool.shutdown(wait=True)

    def __enter__(self) -> ProcessQueryParallel:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
