# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from typing import Any, Self, TypeVar, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Recipe, RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models.base import ICLModel, _categorical_mask
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema

T = TypeVar("T")


class EnsembleParallel:
    """Run independent ensemble members on explicitly placed model replicas.

    Preprocessing and output reduction run once on the input device. Member
    Member ``i`` retains its cache on replica ``i % len(replicas)``.
    Calls are synchronous and must not overlap on the same executor. Replicas
    must have identical weights and remain unchanged while fitted.

    Args:
        replicas: Distinct evaluation models, normally one per CUDA device.
            The caller owns their construction and checkpoint loading.
    """

    def __init__(self, replicas: Sequence[ICLModel]) -> None:
        if len({id(model) for model in replicas}) != len(replicas):
            raise ValueError("Each replica must be a distinct model")
        if any(model.training for model in replicas):
            raise ValueError("Replicas must be in evaluation mode")
        self.replicas = tuple(replicas)
        self.devices = tuple(
            next(model.parameters()).device for model in replicas
        )
        self._workers = tuple(
            ThreadPoolExecutor(max_workers=1) for _ in replicas
        )
        self._streams = tuple(
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
            for device in self.devices
        )
        self._execution: RecipeExecution | None = None
        self._caches: list[Cache] = []
        self._kwargs: dict[str, Any] = {}
        self.member_seeds: tuple[int, ...] = ()

    @property
    def cache_bytes(self) -> tuple[int, ...]:
        """Tensor cache bytes per replica, excluding model weights."""
        return tuple(
            sum(
                cache.size() for cache in self._caches[i :: len(self.replicas)]
            )
            for i in range(len(self.replicas))
        )

    def _dispatch(
        self,
        operation: Callable[[int, ICLModel, torch.device], T],
        count: int,
        source: torch.device,
    ) -> list[T]:
        ready = None
        if source.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(source))
        autocast = {
            device.type: (
                torch.is_autocast_enabled(device.type),
                torch.get_autocast_dtype(device.type),
            )
            for device in self.devices
        }

        def run(worker_id: int) -> list[tuple[int, T]]:
            device = self.devices[worker_id]
            stream = self._streams[worker_id]
            enabled, dtype = autocast[device.type]
            with (
                torch.cuda.device(device)
                if stream is not None
                else nullcontext(),
                torch.cuda.stream(stream)
                if stream is not None
                else nullcontext(),
                torch.inference_mode(),
                torch.autocast(device.type, enabled=enabled, dtype=dtype),
            ):
                if ready is not None:
                    if stream is None:
                        ready.synchronize()
                    else:
                        stream.wait_event(ready)
                try:
                    return [
                        (i, operation(i, self.replicas[worker_id], device))
                        for i in range(worker_id, count, len(self.replicas))
                    ]
                finally:
                    # Synchronous API boundary also drains failed work.
                    if stream is not None:
                        stream.synchronize()

        futures = [
            worker.submit(run, i) for i, worker in enumerate(self._workers)
        ]
        wait(futures)
        outputs = [item for future in futures for item in future.result()]
        return [
            value for _, value in sorted(outputs, key=lambda item: item[0])
        ]

    def fit(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        generator: torch.Generator | None = None,
        member_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Fit shared preprocessing and device-resident member caches.

        ``generator`` controls preprocessing. Model member ``i`` uses a local
        generator seeded with ``member_seed + i``, independent of placement.
        This deliberately separates preprocessing and model randomness; use a
        one-replica executor for the matching serial stochastic baseline.
        Model kwargs are saved and reused on prediction.
        """
        self.clear()
        execution = RecipeExecution(
            self.replicas[0].default_recipe()
            if recipe is None
            else copy.deepcopy(recipe)
        )
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            contexts = execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                generator=generator,
            )
        seeds = tuple(member_seed + i for i in range(len(contexts)))

        def fit_member(i: int, model: ICLModel, device: torch.device) -> Cache:
            context = contexts[i]
            context = MemberContext(
                x=context.x.to(device, non_blocking=True),
                y=context.y.to(device, non_blocking=True),
                related_tables=(
                    None
                    if context.related_tables is None
                    else context.related_tables.to(device, non_blocking=True)
                ),
                input_stypes=context.input_stypes,
            )
            context = model._prepare_context(context, ())
            mask = _categorical_mask([context])
            cache = Cache(
                x_schema=context.x.schema,
                related_tables_schema=(
                    None
                    if context.related_tables is None
                    else context.related_tables.schema
                ),
                classes=(
                    context.y.categorical.categories[0]
                    if context.y.categorical.size(-1)
                    else None
                ),
                categorical_mask=mask,
            )
            model._forward(
                x_context=context.x,
                y_context=context.y,
                x_query=None,
                related_context_tables=context.related_tables,
                related_query_tables=None,
                cache=cache,
                generator=torch.Generator(device=device).manual_seed(seeds[i]),
                categorical_mask=mask,
                **kwargs,
            )
            return cache.freeze()

        caches = self._dispatch(fit_member, len(contexts), x.device)
        self._caches = caches
        self.member_seeds = seeds
        self._kwargs = kwargs
        self._execution = execution

    def predict(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        """Predict query rows, then combine outputs in logical member order."""
        if self._execution is None:
            raise RuntimeError("Call fit before predict")
        schema = cast(
            RelatedTablesSchema | None,
            self._caches[0]["related_tables_schema"],
        )
        if related_tables is not None and schema is not None:
            related_tables = related_tables.select_tables(tables=schema.tables)
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            queries = self._execution.transform(x, related_tables)

        def predict_member(
            i: int, model: ICLModel, device: torch.device
        ) -> TableTensor:
            query, cache = queries[i], self._caches[i]
            query = MemberQuery(
                x=query.x.to(device, non_blocking=True),
                related_tables=(
                    None
                    if query.related_tables is None
                    else query.related_tables.to(device, non_blocking=True)
                ),
            )
            query = model._prepare_query(
                query=query,
                x_schema=cast(TableSchema, cache["x_schema"]),
                related_tables_schema=cast(
                    RelatedTablesSchema | None, cache["related_tables_schema"]
                ),
                callbacks=(),
            )
            output = model._forward_batch(
                contexts=None,
                queries=[query],
                cache=cache,
                categorical_mask=cast(Tensor, cache["categorical_mask"]),
                class_values=None,
                callbacks=(),
                requires_grad=False,
                generator=None,
                **self._kwargs,
            )[0]
            # The worker synchronizes before this tensor is consumed or freed.
            return output.to(x.device, non_blocking=True)

        outputs = self._dispatch(predict_member, len(queries), x.device)
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            if self._caches[0]["classes"] is None:
                outputs = list(
                    self._execution.inverse_transform_target(outputs)
                )
            return self._execution.transform_output(outputs)

    def clear(self) -> None:
        """Release fitted caches and preprocessing state."""
        self._execution = None
        self._caches = []
        self.member_seeds = ()
        self._kwargs = {}

    def close(self) -> None:
        """Join worker threads and release fitted state."""
        for worker in self._workers:
            worker.shutdown(wait=True)
        self.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
