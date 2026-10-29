# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit owner-fit and original-batch execution for KumoTabular."""

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from types import MethodType
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.execution._collectives import Exchange
from sdm.execution._tabular_cache import OwnedCache
from sdm.execution._tabular_icl import fit_icl, predict_icl, workspace_bytes
from sdm.execution._tabular_rows import Rows, bounds
from sdm.models import KumoTabular
from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.nn import TransformerBlock
from sdm.processing.execution import MemberQuery, RecipeExecution
from sdm.tensor.table import TableSchema

_RETIRED: list["TabularExecution"] = []


def _tensor_identity(value: Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        value.data_ptr(),
        value._version,
        value.shape,
        value.stride(),
        value.dtype,
        value.device,
    )


def _model_identity(model: KumoTabular) -> tuple[Any, ...]:
    modules = torch.nn.modules.module
    if modules._global_forward_hooks or modules._global_forward_pre_hooks:
        raise ValueError("Execution does not support global forward hooks")
    for module in model.modules():
        if (
            module.training
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._compiled_call_impl is not None
            or any(
                key in vars(module)
                for key in ("forward", "_call_impl", "__call__")
            )
        ):
            raise ValueError(
                "Execution requires unmodified evaluation modules "
                "without hooks"
            )
        if not cast(MethodType, module.forward).__func__.__module__.startswith(
            ("sdm.", "torch.nn.")
        ):
            raise ValueError(
                "Execution does not support custom module dispatch"
            )
    return (
        tuple(
            (id(module), cast(MethodType, module.forward).__func__)
            for module in model.modules()
        ),
        tuple(
            _tensor_identity(value)
            for value in (*model.parameters(), *model.buffers())
        ),
    )


def _check_output(recipe: Recipe) -> None:
    # Waves prepare several inputs before finishing their outputs. Known
    # deterministic output steps preserve the original RNG ordering.
    allowed = (
        sp.Identity,
        sp.Sequential,
        sp.EnsembleProcessorAdapter,
        sp.AverageEstimators,
        sp.Softmax,
        sp.SortQuantiles,
        sp.TaskDispatch,
        torch.nn.ModuleDict,
        torch.nn.ModuleList,
    )
    if any(type(module) not in allowed for module in recipe.output.modules()):
        raise ValueError(
            "Execution requires deterministic built-in output processors"
        )


class TabularExecution:
    """Run an explicitly owned, eager E1 tabular generation.

    Construct with :func:`tabular_execution`. All ranks call every operation;
    only group rank zero supplies inputs and receives public outputs. Groups,
    model loading and process lifetime remain the caller's responsibility.
    Every rank must load identical weights and use the same operation order.
    The model is exclusively borrowed until healthy close. Do not mutate it,
    recipes or cached tensors, or launch foreign-stream work during a call.
    Operations must run on the constructing thread. Boundary version checks
    are not a defense against arbitrary ``.data`` or foreign-stream mutation.
    A failed operation permanently retires this executor and retains its
    owners; failed CUDA/collectives require process retirement, not recovery.
    Original-wave replay requires deterministic built-in output processors.
    Fit retains row-owned cells/ICL KV and column-owned frontend KV, but full
    ICL state and a native column window can still span the entire context.
    """

    def __init__(
        self,
        model: KumoTabular,
        task: Task,
        exchange: Exchange,
        column_staging_bytes: int,
        attention_workspace_bytes: int,
        fit_head_gather_window: int | None = None,
        owner_local_frontend: bool = False,
        shard_row_workspace: bool = False,
        precision: Literal["float32", "float16"] = "float32",
        gathered_kv_cache_bytes: int = 0,
    ) -> None:
        self.model, self.task, self.exchange = model, task, exchange
        self._core = cast(_KumoTabular, model.models[task])
        self.device = next(self._core.parameters()).device
        self.column_staging_bytes = column_staging_bytes
        self.attention_workspace_bytes = attention_workspace_bytes
        self.fit_head_gather_window = fit_head_gather_window
        self.owner_local_frontend = owner_local_frontend
        self.shard_row_workspace = shard_row_workspace
        self.precision = precision
        self.gathered_kv_cache_bytes = gathered_kv_cache_bytes
        self._lock = threading.Lock()
        self._owner_thread = threading.get_ident()
        self._thread: int | None = None
        self._operation: str | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._root_cache: Cache | None = None
        self._batch_cache: Cache | None = None
        self._row_cache: OwnedCache | None = None
        self._icl_cache: OwnedCache | None = None
        self._result: Tensor | None = None
        self._fit_started = False
        self._kv_decision_made = False
        self._identity = exchange.attempt(self._admit)
        signature = tuple(
            (value.shape, value.dtype) for value in self._core.parameters()
        )
        policies = exchange.records(
            (
                task,
                column_staging_bytes,
                attention_workspace_bytes,
                fit_head_gather_window,
                owner_local_frontend,
                shard_row_workspace,
                precision,
                gathered_kv_cache_bytes,
                signature,
            )
        )
        if any(policy != policies[0] for policy in policies):
            raise ValueError("Execution policies differ between ranks")
        self._cache_identity: tuple[Any, ...] | None = None
        self._policy_identity = self._policy()
        model._execution = self

    def _policy(self) -> tuple[Any, ...]:
        return (
            id(self.exchange.group),
            id(self.exchange.control_group),
            self.exchange.rank,
            self.exchange.world,
            self.exchange.members,
            self.task,
            self.device,
            self.column_staging_bytes,
            self.attention_workspace_bytes,
            self.fit_head_gather_window,
            self.owner_local_frontend,
            self.shard_row_workspace,
            self.precision,
            self.gathered_kv_cache_bytes,
        )

    def _check_precision(self) -> None:
        if self.precision not in ("float32", "float16"):
            raise ValueError("Execution precision must be float32 or float16")
        if torch.is_autocast_enabled(self.device.type):
            if self.precision == "float32":
                raise ValueError("Execution does not support autocast")
            if torch.get_autocast_dtype(self.device.type) != torch.float16:
                raise ValueError(
                    "Ambient autocast differs from execution precision"
                )

    def _precision_scope(self) -> AbstractContextManager:
        # Every worker owns the same scope, even without caller autocast.
        # Original recipe and public-processing islands disable it.
        return (
            nullcontext()
            if self.precision == "float32"
            else torch.autocast(
                self.device.type, dtype=torch.float16, cache_enabled=True
            )
        )

    def _admit(self) -> tuple[Any, ...]:
        if torch.is_inference_mode_enabled():
            raise ValueError("Execution does not support inference_mode")
        if self.shard_row_workspace and (
            not self.owner_local_frontend
            or self.exchange.world not in (1, 2, 4, 8)
        ):
            raise ValueError(
                "Sharded row workspace requires owner-local frontend "
                "and P1/2/4/8"
            )
        if self.fit_head_gather_window is not None and (
            type(self.fit_head_gather_window) is not int
            or self.fit_head_gather_window < 1
        ):
            raise ValueError(
                "Fit head gather window must be a positive integer"
            )
        if (
            type(self.gathered_kv_cache_bytes) is not int
            or self.gathered_kv_cache_bytes < 0
        ):
            raise ValueError(
                "Gathered KV cache budget must be a nonnegative integer"
            )
        if (
            type(self.model) is not KumoTabular
            or type(self._core) is not _KumoTabular
        ):
            raise ValueError(
                "Execution requires a stock KumoTabular task core"
            )
        if self.model._execution is not None or self.model._cache is not None:
            raise ValueError("Execution requires an unfitted, unleased model")
        if self.device.type not in ("cpu", "cuda") or dist.get_backend(
            self.exchange.group
        ) != ("nccl" if self.device.type == "cuda" else "gloo"):
            raise ValueError("Execution requires CUDA/NCCL or CPU/Gloo")
        if any(
            value.dtype != torch.float32 or value.device != self.device
            for value in self._core.parameters()
        ):
            raise ValueError(
                "Execution requires one FP32 device per task core"
            )
        self._check_precision()
        if any(
            cast(TransformerBlock, layer).attn.num_query_heads
            % self.exchange.world
            or cast(TransformerBlock, layer).attn.num_key_value_heads
            % self.exchange.world
            for layer in self._core.icl_block.layers
        ):
            raise ValueError("ICL fit heads must divide evenly across owners")
        return _model_identity(self.model)

    def _check_model_call(self, operation: str) -> None:
        if self._failure is not None:
            raise self._failure
        if self._thread != threading.get_ident() or self._operation is None:
            raise RuntimeError(
                "Use the owning tabular executor while its lease is active"
            )
        if operation not in (self._operation, "clear"):
            raise RuntimeError(
                "The model operation differs from its execution lease"
            )

    def _fence(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _retire(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            _RETIRED.append(self)

    def _cache_state(self) -> tuple[Any, ...]:
        return (
            id(self.model._cache),
            tuple((key, id(value)) for key, value in self.model._cache.items())
            if self.model._cache is not None
            else (),
            id(self._row_cache),
            id(self._icl_cache),
            tuple(
                _tensor_identity(value)
                for cache in (self._row_cache, self._icl_cache)
                if cache is not None
                for value in cache._tensors()
            ),
            self._icl_cache.replica_state()
            if self._icl_cache is not None
            else None,
        )

    @contextmanager
    def _call(self, operation: str) -> Iterator[None]:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("Tabular execution is closed")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Tabular execution belongs to another thread")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Tabular execution operations cannot overlap")
        self._thread, self._operation = threading.get_ident(), operation
        try:

            def validate() -> None:
                if operation in ("fit", "predict") and (
                    torch.is_inference_mode_enabled()
                ):
                    raise ValueError(
                        "Execution does not support inference_mode"
                    )
                if (
                    self.model._execution is not self
                    or _model_identity(self.model) != self._identity
                ):
                    raise RuntimeError(
                        "The borrowed model changed; this execution is stale"
                    )
                self._check_precision()
                if self._policy() != self._policy_identity:
                    raise RuntimeError(
                        "Execution policy changed after admission"
                    )
                if (
                    self._cache_identity is not None
                    and self._cache_state() != self._cache_identity
                ):
                    raise RuntimeError(
                        "The fitted cache changed outside execution"
                    )

            self.exchange.attempt(validate)
            with self._precision_scope():
                yield
            if operation in ("fit", "predict"):
                self.exchange.attempt(self._fence)
        except BaseException as error:
            # Traceback and executor keep submitted inputs, caches and owners
            # alive. No cleanup fence can turn uncertain work into reuse.
            self._retire(error)
            raise
        finally:
            self._thread, self._operation = None, None
            self._lock.release()

    def _prepared_forward(
        self, forward: Callable[..., TableTensor], **kwargs: Any
    ) -> TableTensor:
        cache = kwargs["cache"]
        x = (
            kwargs["x_context"]
            if kwargs["x_query"] is None
            else kwargs["x_query"]
        )
        if x.ndim != 2 or cache is None:
            raise ValueError(
                "Execution supports one unbatched estimator with a cache"
            )
        classes = cache["classes"]
        actual = Task.regression if classes is None else Task.classification
        if actual != self.task:
            raise ValueError(
                "Prepared task differs from the selected execution task"
            )
        if classes is not None and len(classes) > self.model.ecoc.max_classes:
            raise ValueError("Execution does not support expanded ECOC tasks")
        self._batch_cache = cache
        return forward(**kwargs)

    def _run_core(
        self, task: Task, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> Tensor:
        if self._result is not None:
            result, self._result = self._result, None
            return result
        if task != self.task or model is not self._core:
            raise ValueError(
                "Prepared core differs from the selected execution core"
            )
        x = kwargs.get("x", args[0] if args else None)
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        if x.dtype != torch.float32 or x.device != self.device:
            raise ValueError("Prepared inputs must match the FP32 core device")
        # Finish root-only allocations before announcing the tensor phase.
        # Otherwise an allocation error could strand peers in a broadcast.
        x, y, mask = (
            x.contiguous(),
            y.contiguous(),
            kwargs["categorical_mask"].contiguous(),
        )
        self._fit_started = True
        self.exchange.metadata((None, x.shape, y.dtype))
        self.exchange.agree(None)  # Peers allocate the matching destinations.
        self._fit_core(x, y, mask)
        return x.new_empty((0, model.icl_block.head[-1].out_features))

    def _fit_core(self, x: Tensor, y: Tensor, mask: Tensor) -> None:
        exchange = self.exchange
        exchange.broadcast(x)
        exchange.broadcast(y)
        exchange.broadcast(mask)
        rows = exchange.attempt(
            lambda: Rows(
                self._core.row_embedding,
                x,
                y,
                mask,
                exchange,
                self.column_staging_bytes,
                owner_local_frontend=self.owner_local_frontend,
                shard_row_workspace=self.shard_row_workspace,
            )
        )
        self._row_cache = OwnedCache(
            exchange, bounds(x.size(1), exchange.world), 0, "row_embedding."
        )
        embedded = exchange.attempt(
            lambda: self._core.row_embedding(
                x[rows.start : rows.stop],
                y[rows.start : rows.stop],
                mask,
                cache=self._row_cache,
                _runner=rows,
            )
        )
        embedded = exchange.attempt(lambda: self._core.row_project(embedded))
        self._icl_cache = OwnedCache(exchange, rows.lengths, 0, "icl_block.")
        fit_icl(
            self._core.icl_block,
            embedded,
            y,
            self._icl_cache,
            exchange,
            self.fit_head_gather_window,
        )
        self._row_cache.freeze()
        self._icl_cache.freeze()
        # Successful work no longer needs callbacks that keep Rows in a cycle.
        # Failures retain the original traceback/owners instead of cleaning up.
        rows.handlers.clear()

    def fit(
        self,
        x: Tensor | TableTensor | None,
        y: Tensor | TableTensor | None,
        *,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        """Fit one estimator; only rank zero provides context and recipe."""
        with self._call("fit"), torch.no_grad():
            self.exchange.attempt(self._fence)
            self._release()
            self._fit_started = False

            def inputs() -> None:
                if self.exchange.rank != 0 and (
                    x is not None or y is not None
                ):
                    raise ValueError("Only root supplies fit inputs")
                if any(
                    value is not None and value.requires_grad
                    for value in (x, y)
                ):
                    raise ValueError(
                        "Execution does not support autograd inputs"
                    )

            self.exchange.attempt(inputs)

            def execute() -> None:
                if self.exchange.rank == 0:
                    try:
                        if (
                            x is None
                            or y is None
                            or x.ndim != 2
                            or y.ndim != 2
                        ):
                            raise ValueError(
                                "Root must supply two-dimensional fit inputs"
                            )
                        chosen = (
                            self.model.default_recipe()
                            if recipe is None
                            else recipe
                        )
                        _check_output(chosen)
                        self.model.fit(
                            x,
                            y,
                            recipe=chosen,
                            num_estimators=1,
                            generator=generator,
                        )
                        self._root_cache = self.model._cache
                    except BaseException as error:
                        if not self._fit_started:
                            self.exchange.metadata((str(error), None, None))
                        raise
                else:
                    error, shape, dtype = self.exchange.metadata(None)
                    if error is not None:
                        raise ValueError(error)
                    inputs = self.exchange.attempt(
                        lambda: (
                            torch.empty(
                                shape, dtype=torch.float32, device=self.device
                            ),
                            torch.empty(
                                shape[0], dtype=dtype, device=self.device
                            ),
                            torch.empty(
                                shape[1], dtype=torch.bool, device=self.device
                            ),
                        )
                    )
                    self._fit_core(*inputs)

            self.exchange.attempt(execute)
            self._cache_identity = self._cache_state()

    def _prepare(
        self, batch: Tensor | TableTensor
    ) -> tuple[MemberQuery, Cache]:
        root = self._root_cache
        assert root is not None
        cache = cast(Cache, root[0])
        execution = cast(RecipeExecution, root["recipe_execution"])
        with torch.amp.autocast(batch.device.type, enabled=False):
            queries = execution.transform(batch, None)
        if len(queries) != 1:
            raise ValueError(
                "Execution supports exactly one prepared query member"
            )
        query = self.model._prepare_query(
            queries[0],
            cast(tuple[TableSchema, ...], cache["x_schemas"])[0],
            None,
            (),
        )
        if query.x.ndim != 2 or query.x.numerical.size(0) != batch.size(0):
            raise ValueError(
                "Preparation must preserve each original query batch"
            )
        return query, cache

    def _finish(
        self, prepared: tuple[MemberQuery, Cache], raw: Tensor
    ) -> TableTensor:
        query, cache = prepared
        self._result = raw
        outputs = self.model._forward_batch(
            contexts=None,
            queries=[query],
            cache=cache,
            categorical_mask=cast(Tensor, cache["categorical_mask"]),
            class_values=cast(
                Sequence[tuple[Any, ...] | None], cache["class_values"]
            ),
            callbacks=(),
            requires_grad=False,
            generator=None,
        )
        if self._result is not None:
            raise RuntimeError(
                "The public output did not consume its computed batch"
            )
        assert self._root_cache is not None
        execution = cast(RecipeExecution, self._root_cache["recipe_execution"])
        with torch.amp.autocast(raw.device.type, enabled=False):
            if self.task == Task.regression:
                outputs = list(execution.inverse_transform_target(outputs))
            return execution.transform_output(outputs)

    def predict(self, x: Tensor | TableTensor | None) -> TableTensor | None:
        """Predict one original batch without splitting or rebatching it."""
        values = self.predict_batches(
            [cast(Tensor, x)] if self.exchange.rank == 0 else None
        )
        return values[0] if values is not None else None

    def predict_batches(
        self,
        batches: Sequence[Tensor | TableTensor] | None,
    ) -> list[TableTensor] | None:
        """Predict finite original batches, returning their original order."""
        with self._call("predict"), torch.no_grad():

            def plan() -> list[list[int]] | None:
                if self._icl_cache is None:
                    raise RuntimeError("Tabular execution is not fitted")
                if self.exchange.rank != 0:
                    if batches is not None:
                        raise ValueError("Only root supplies query batches")
                    return None
                if not isinstance(batches, Sequence):
                    raise ValueError(
                        "Root must provide a finite batch sequence"
                    )
                waves, previous = [], None
                for index, batch in enumerate(batches):
                    if batch.ndim != 2 or batch.size(0) == 0:
                        raise ValueError(
                            "Original query batches must be nonempty "
                            "and two-dimensional"
                        )
                    geometry = (batch.shape, batch.dtype, batch.device)
                    if batch.requires_grad:
                        raise ValueError(
                            "Execution does not support autograd inputs"
                        )
                    if (
                        not waves
                        or len(waves[-1]) == self.exchange.world
                        or geometry != previous
                    ):
                        waves.append([])
                    waves[-1].append(index)
                    previous = geometry
                return waves

            waves = self.exchange.metadata(self.exchange.attempt(plan))
            if waves:
                self._materialize_icl()
            outputs = []
            for wave in waves:
                outputs.extend(self._predict_wave(wave, batches))
            return outputs if self.exchange.rank == 0 else None

    def _materialize_icl(self) -> None:
        cache = self._icl_cache
        assert cache is not None
        if (
            self._kv_decision_made
            or self.gathered_kv_cache_bytes == 0
            or self.exchange.world == 1
        ):
            return
        layout, needed = self.exchange.attempt(cache.replica_plan)
        decision = (layout, needed <= self.gathered_kv_cache_bytes)
        decisions = self.exchange.records(decision)
        if any(other != decision for other in decisions):
            raise RuntimeError("ICL KV replica plans differ between ranks")
        self._kv_decision_made = True
        if not decision[1]:
            return
        cache.materialize_replicas(layout)
        # The owned slices and complete replicas become one boundary identity.
        self._cache_identity = self._cache_state()

    def _predict_wave(
        self, wave: list[int], batches: Sequence[Tensor | TableTensor] | None
    ) -> list[TableTensor]:
        outputs: list[TableTensor] = []
        prepared = self.exchange.attempt(
            lambda: (
                [
                    self._prepare(cast(Sequence[Tensor], batches)[i])
                    for i in wave
                ]
                if self.exchange.rank == 0
                else None
            )
        )

        def check_geometry() -> None:
            if prepared is None:
                return
            expected = (
                prepared[0][0].x.numerical.shape,
                cast(Tensor, prepared[0][1]["categorical_mask"]).shape,
            )
            if any(
                (
                    q.x.numerical.shape,
                    cast(Tensor, c["categorical_mask"]).shape,
                )
                != expected
                or q.x.numerical.dtype != torch.float32
                or q.x.device != self.device
                for q, c in prepared
            ):
                raise ValueError(
                    "Prepared query geometries differ within a wave"
                )

        self.exchange.attempt(check_geometry)
        metadata = None
        if prepared is not None:
            metadata = (
                prepared[0][0].x.numerical.shape,
                cast(Tensor, prepared[0][1]["categorical_mask"]).shape,
            )
        shape, mask_shape = self.exchange.metadata(metadata)
        assert self._icl_cache is not None
        if (
            workspace_bytes(self._core.icl_block, shape[0], self._icl_cache)
            > self.attention_workspace_bytes
        ):
            raise ValueError(
                "Original query exceeds the attention workspace budget"
            )
        local, mask = self.exchange.attempt(
            lambda: (
                torch.empty(shape, dtype=torch.float32, device=self.device),
                torch.empty(mask_shape, dtype=torch.bool, device=self.device),
            )
        )
        sources = (
            prepared + [prepared[-1]] * (self.exchange.world - len(prepared))
            if prepared is not None
            else None
        )
        packed = self.exchange.attempt(
            lambda: (
                (
                    [q.x.numerical.contiguous() for q, _ in sources],
                    [
                        cast(Tensor, c["categorical_mask"]).contiguous()
                        for _, c in sources
                    ],
                )
                if sources is not None
                else (None, None)
            )
        )
        dist.scatter(
            local,
            packed[0],
            src=self.exchange.members[0],
            group=self.exchange.group,
        )
        dist.scatter(
            mask,
            packed[1],
            src=self.exchange.members[0],
            group=self.exchange.group,
        )
        empty = self.exchange.attempt(
            lambda: local.new_empty(
                (0,),
                dtype=torch.int64
                if self.task == Task.classification
                else local.dtype,
            )
        )
        state = self.exchange.attempt(
            lambda: self._core.row_embedding(
                local, empty, mask, cache=self._row_cache
            )
        )
        state = self.exchange.attempt(lambda: self._core.row_project(state))
        raw = predict_icl(self._core.icl_block, state, self._icl_cache)
        raw = self.exchange.attempt(raw.contiguous)
        received = self.exchange.attempt(
            lambda: (
                [torch.empty_like(raw) for _ in range(self.exchange.world)]
                if self.exchange.rank == 0
                else None
            )
        )
        dist.gather(
            raw,
            received,
            dst=self.exchange.members[0],
            group=self.exchange.group,
        )

        def finish() -> None:
            if prepared is not None:
                assert received is not None
                outputs.extend(
                    self._finish(item, value)
                    for item, value in zip(prepared, received)
                )

        self.exchange.attempt(finish)
        return outputs

    def _release(self) -> None:
        self.model.clear()
        self._root_cache = self._batch_cache = None
        self._row_cache = self._icl_cache = None
        self._cache_identity = None
        self._kv_decision_made = False

    def clear(self) -> None:
        """Fence and release the fitted generation, keeping the model lease."""
        with self._call("clear"):
            self.exchange.attempt(self._fence)
            self._release()

    def close(self) -> None:
        """Fence and release the lease; never destroy caller-owned groups."""
        if self._closed:
            return
        with self._call("close"):
            self.exchange.attempt(self._fence)
            self._release()
        self.model._execution = None
        self._closed = True


def tabular_execution(
    model: KumoTabular,
    *,
    task: TaskLike,
    group: dist.ProcessGroup,
    control_group: dist.ProcessGroup,
    column_staging_bytes: int = 8 << 30,
    attention_workspace_bytes: int = 12 << 30,
    fit_head_gather_window: int | None = None,
    owner_local_frontend: bool = False,
    shard_row_workspace: bool = False,
    precision: Literal["float32", "float16"] = "float32",
    gathered_kv_cache_bytes: int = 0,
) -> TabularExecution:
    """Borrow a loaded model for explicit original-wave execution.

    Args:
        model: Caller-owned, unfitted FP32 evaluation KumoTabular model.
        task: Selected task; classification must fit native class capacity.
        group: Caller-owned tensor process group, one rank per device.
        control_group: Caller-owned Gloo group with identical ordered members.
        column_staging_bytes: Component bound for native column transport.
        attention_workspace_bytes: Logical cached-attention scratch admission.
            Neither component budget bounds total allocator/device memory.
        fit_head_gather_window: Optional limit on fit head gathers enqueued
            ahead of completed gathered outputs. ``None`` preserves the
            original asynchronous enqueue behavior.
        owner_local_frontend: Experimental local intersections of native cell
            and row windows. Changes projection geometry; trained CUDA
            numerical equivalence and capacity require separate evidence.
        shard_row_workspace: Further bound owner-local row-block calls to
            max(1, native row limit // owners). Requires owner-local frontend
            and P1/2/4/8. Default False retains the native row-window budget.
        precision: Neural execution precision. Default float32 rejects
            ambient autocast. Explicit float16 owns matching autocast scopes
            on every rank; weights, prepared inputs and original processing
            islands remain FP32. This is not half-stored model support.
        gathered_kv_cache_bytes: Optional per-rank retained-payload budget
            for a whole generation's gathered ICL K/V. Default zero keeps
            per-layer gathers. A positive budget is used only when all full
            layers fit; owned slices remain resident alongside replicas.
            Each rank retains O(context rows) replica payload, independent
            of group size. Single-rank execution ignores this option.
            This does not bound transient collective workspace or total
            allocator/device memory.

    All ranks construct the executor and call its methods in the same order::

        run = tabular_execution(model, task="regression", group=group,
                                control_group=control_group)
        run.fit(x if root else None, y if root else None)
        outputs = run.predict_batches(batches if root else None)
        run.close()

    Each batch keeps its original shape and public recipe postprocessing.
    This scope is eager with one unbatched estimator, no callbacks,
    no outer ``torch.inference_mode()`` (tensor version fences are required),
    and no ECOC expansion beyond the native class capacity. Groups and model
    loading are caller-owned; a healthy close permits ordinary model reuse.
    """
    return TabularExecution(
        model,
        Task(task),
        Exchange(group, control_group),
        column_staging_bytes,
        attention_workspace_bytes,
        fit_head_gather_window,
        owner_local_frontend,
        shard_row_workspace,
        precision,
        gathered_kv_cache_bytes,
    )
