# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent process ensemble execution with shared parent preprocessing."""

from __future__ import annotations

import copy
import multiprocessing
import os
import resource
import sys
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Recipe, RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models.base import _categorical_mask
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)

_model: ICLModel
_device: torch.device
_dtype: torch.dtype | None
_caches: dict[int, Cache]
_kwargs: dict[str, Any]


def _initialize(
    model: ICLModel, device: torch.device, dtype: torch.dtype | None
) -> None:
    global _model, _device, _dtype, _caches, _kwargs
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _model, _device, _dtype = model.to(device).eval(), device, dtype
    _caches, _kwargs = {}, {}


def _clear() -> None:
    global _caches, _kwargs
    _caches, _kwargs = {}, {}


def _memory(reset_peak: bool = False) -> dict[str, Any]:
    seen = set()
    cache_bytes = 0
    for cache in _caches.values():
        for tensor in cache._tensors():
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                cache_bytes += storage.nbytes()
    result = {
        "pid": os.getpid(),
        "device": str(_device),
        "cache_storage_bytes": cache_bytes,
        "max_cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
        result.update(
            allocated_bytes=torch.cuda.memory_allocated(_device),
            reserved_bytes=torch.cuda.memory_reserved(_device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(_device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(_device),
        )
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(_device)
    return result


def _fit(
    members: list[tuple[int, MemberContext]],
    member_seed: int,
    kwargs: dict[str, Any],
) -> None:
    global _kwargs
    _clear()
    _kwargs = kwargs
    with (
        torch.inference_mode(),
        torch.autocast(_device.type, dtype=_dtype, enabled=_dtype is not None),
    ):
        for member_id, context in members:
            context = MemberContext(
                x=context.x.to(_device),
                y=context.y.to(_device),
                related_tables=(
                    None
                    if context.related_tables is None
                    else context.related_tables.to(_device)
                ),
                input_stypes=context.input_stypes,
            )
            context = _model._prepare_context(context, ())
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
            _model._forward(
                x_context=context.x,
                y_context=context.y,
                x_query=None,
                related_context_tables=context.related_tables,
                related_query_tables=None,
                cache=cache,
                generator=torch.Generator(device=_device).manual_seed(
                    member_seed + member_id
                ),
                categorical_mask=mask,
                **kwargs,
            )
            _caches[member_id] = cache.freeze()
        if _device.type == "cuda":
            torch.cuda.synchronize(_device)


def _predict(
    members: list[tuple[int, MemberQuery]],
) -> list[tuple[int, TableTensor]]:
    outputs = []
    with (
        torch.inference_mode(),
        torch.autocast(_device.type, dtype=_dtype, enabled=_dtype is not None),
    ):
        for member_id, query in members:
            cache = _caches[member_id]
            query = MemberQuery(
                x=query.x.to(_device),
                related_tables=(
                    None
                    if query.related_tables is None
                    else query.related_tables.to(_device)
                ),
            )
            query = _model._prepare_query(
                query=query,
                x_schema=cache["x_schema"],
                related_tables_schema=cache["related_tables_schema"],
                callbacks=(),
            )
            output = _model._forward_batch(
                contexts=None,
                queries=[query],
                cache=cache,
                categorical_mask=cache["categorical_mask"],
                class_values=None,
                callbacks=(),
                requires_grad=False,
                generator=None,
                **_kwargs,
            )[0]
            outputs.append((member_id, output.cpu()))
    return outputs


class ProcessEnsembleParallel:
    """Use persistent independent Python processes for ensemble members.

    Args:
        replicas: Identical models, moved to CPU for spawn transport.
        dtype: Explicit child autocast dtype, or None for FP32.
        member_seed: Placement-independent first model-member seed.

    Preprocessing and final reduction remain in the parent. Transformed tables
    cross process boundaries through CPU shared memory. This avoids
    CUDA IPC lifetime assumptions but includes device/host transfer overhead.
    """

    def __init__(
        self,
        replicas: Sequence[ICLModel],
        *,
        dtype: torch.dtype | None = None,
        member_seed: int = 0,
    ) -> None:
        self.devices = tuple(
            next(model.parameters()).device for model in replicas
        )
        self._recipe = replicas[0].default_recipe
        self._member_seed = member_seed
        self._execution: RecipeExecution | None = None
        self._schema: Any = None
        self._regression = False
        self.member_seeds: tuple[int, ...] = ()
        self._pools = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(model.cpu(), device, dtype),
            )
            for model, device in zip(replicas, self.devices, strict=True)
        ]
        # Parent replicas were moved to CPU; release their inactive CUDA
        # allocator blocks before children allocate their own parameter copies.
        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        try:
            self.memory()  # Include process/model startup in load timing.
        except BaseException:
            self.close()
            raise

    def memory(self, *, reset_peak: bool = False) -> list[dict[str, Any]]:
        """Read child allocator and unique cache storage bytes."""
        futures = [pool.submit(_memory, reset_peak) for pool in self._pools]
        return [future.result() for future in futures]

    def clear(self) -> None:
        """Discard all member caches and preprocessing state."""
        self._execution = None
        self.member_seeds = ()
        futures = [pool.submit(_clear) for pool in self._pools]
        wait(futures)
        for future in futures:
            future.result()

    def fit(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        generator: torch.Generator | None = None,
        member_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit one recipe and cache assigned members in each rank."""
        self.clear()
        execution = RecipeExecution(
            self._recipe() if recipe is None else copy.deepcopy(recipe)
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
        groups: list[list[tuple[int, MemberContext]]] = [
            [] for _ in self._pools
        ]
        for i, context in enumerate(contexts):
            groups[i % len(groups)].append(
                (
                    i,
                    MemberContext(
                        x=context.x.cpu(),
                        y=context.y.cpu(),
                        related_tables=None
                        if context.related_tables is None
                        else context.related_tables.cpu(),
                        input_stypes=context.input_stypes,
                    ),
                )
            )
        seed = self._member_seed if member_seed is None else member_seed
        futures = [
            pool.submit(_fit, group, seed, kwargs)
            for pool, group in zip(self._pools, groups, strict=True)
        ]
        wait(futures)
        for future in futures:
            future.result()
        self._schema = (
            None
            if contexts[0].related_tables is None
            else contexts[0].related_tables.schema
        )
        self._regression = contexts[0].y.categorical.size(-1) == 0
        self.member_seeds = tuple(seed + i for i in range(len(contexts)))
        self._execution = execution

    def predict(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        """Predict assigned members and reduce their ordered outputs once."""
        if self._execution is None:
            raise RuntimeError("Call fit before predict")
        if related_tables is not None and self._schema is not None:
            related_tables = related_tables.select_tables(
                tables=self._schema.tables
            )
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            queries = self._execution.transform(x, related_tables)
        groups: list[list[tuple[int, MemberQuery]]] = [[] for _ in self._pools]
        for i, query in enumerate(queries):
            groups[i % len(groups)].append(
                (
                    i,
                    MemberQuery(
                        x=query.x.cpu(),
                        related_tables=None
                        if query.related_tables is None
                        else query.related_tables.cpu(),
                    ),
                )
            )
        futures = [
            pool.submit(_predict, group)
            for pool, group in zip(self._pools, groups, strict=True)
        ]
        wait(futures)
        members = sorted(
            (item for future in futures for item in future.result()),
            key=lambda item: item[0],
        )
        with (
            torch.inference_mode(),
            torch.autocast(x.device.type, enabled=False),
        ):
            outputs = [
                cast(TableTensor, output.to(x.device)) for _, output in members
            ]
            if self._regression:
                outputs = list(
                    self._execution.inverse_transform_target(outputs)
                )
            return self._execution.transform_output(outputs)

    def close(self) -> None:
        """Join child processes and release all model/cache allocations."""
        for pool in self._pools:
            pool.shutdown(wait=True)
        self._execution = None


def factory(
    args: Namespace, replicas: Sequence[ICLModel]
) -> ProcessEnsembleParallel:
    """Build process ensemble workers for the shared benchmark harness."""
    return ProcessEnsembleParallel(
        replicas,
        dtype=torch.bfloat16 if args.precision == "bfloat16" else None,
        member_seed=args.seed,
    )
