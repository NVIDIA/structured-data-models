from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Literal, NoReturn, TypeVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.models.base import ICLModel
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema

T = TypeVar("T")


def _serialized(method: Any) -> Any:
    @wraps(method)
    def wrapped(self: EnsembleParallel, *args: Any, **kwargs: Any) -> Any:
        with self._call_lock:
            return method(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class _MemberState:
    worker_id: int
    cache: Cache


@dataclass(frozen=True)
class _FittedState:
    execution: RecipeExecution
    members: tuple[_MemberState, ...]
    kwargs: dict[str, Any]
    is_regression: bool
    member_plan: EnsembleMemberPlan


@dataclass(frozen=True)
class _AutocastState:
    enabled: bool
    dtype: torch.dtype


@dataclass(frozen=True)
class _WorkerOutput:
    member_id: int
    output: TableTensor
    event: torch.cuda.Event | None


@dataclass(frozen=True)
class _MemberPlan:
    seeds: tuple[int | None, ...]
    shared_generator: torch.Generator | None
    evidence: EnsembleMemberPlan


@dataclass(frozen=True)
class _TensorSignature:
    kind: Literal["parameter", "buffer"]
    name: str
    object_id: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    data_ptr: int
    version: int
    requires_grad: bool


@dataclass(frozen=True)
class _ModelSignature:
    object_id: int
    model_type: type[torch.nn.Module]
    training: bool
    device: torch.device
    tensors: tuple[_TensorSignature, ...]


@dataclass(frozen=True)
class EnsembleMemberPlan:
    r"""Placement-independent evidence for a fitted estimator RNG plan."""

    version: int
    policy: Literal["canonical", "member"]
    member_seeds: tuple[int | None, ...]
    generator_state_before_sha256: str | None
    generator_state_after_sha256: str | None
    digest_sha256: str

    def digest_payload(self) -> dict[str, object]:
        r"""Return the canonical JSON payload covered by the digest."""
        return {
            "version": self.version,
            "policy": self.policy,
            "member_seeds": list(self.member_seeds),
            "generator_state_before_sha256": (
                self.generator_state_before_sha256
            ),
            "generator_state_after_sha256": (
                self.generator_state_after_sha256
            ),
        }

    def as_dict(self) -> dict[str, object]:
        r"""Return a fresh JSON-serializable evidence record."""
        return {**self.digest_payload(), "digest_sha256": self.digest_sha256}


class EnsembleParallel(torch.nn.Module):
    r"""Experimentally execute estimator members across model replicas.

    Each replica owns one long-lived worker and must already be placed on its
    target device. Ensemble preprocessing and output processing execute once
    on the caller device, while estimator IDs are assigned to replicas in
    round-robin order.

    Args:
        models: Equivalent, explicit :class:`ICLModel` replicas. CUDA replicas
            must occupy distinct devices.
        rng_policy: Model-core random stream policy. ``"canonical"`` preserves
            ordinary sequential :class:`ICLModel` generator behavior and
            requires exactly one replica. ``"member"`` derives a stable
            generator seed for every logical estimator and requires an
            explicit caller generator during :meth:`forward` and :meth:`fit`.

    .. warning::

        This experimental API keeps fitted estimator caches resident on their
        worker devices. It supports inference-only callbacks and does not
        support serialization of live workers.
    """

    def __init__(
        self,
        models: Sequence[ICLModel],
        *,
        rng_policy: Literal["canonical", "member"] = "canonical",
    ) -> None:
        super().__init__()
        if len(models) == 0:
            raise ValueError("Expected at least one model replica")
        if not all(isinstance(model, ICLModel) for model in models):
            raise TypeError("Expected every model replica to be an ICLModel")
        if len({id(model) for model in models}) != len(models):
            raise ValueError("Expected distinct model replica objects")
        if len({type(model) for model in models}) != 1:
            raise ValueError("Expected model replicas of the same type")
        if any(model.training for model in models):
            raise ValueError("Expected model replicas in evaluation mode")
        if rng_policy == "canonical" and len(models) != 1:
            raise ValueError(
                "The 'canonical' RNG policy requires exactly one model "
                "replica; use 'member' for worker-invariant execution"
            )

        self.models = torch.nn.ModuleList(models)
        self._rng_policy: Literal["canonical", "member"] = rng_policy
        self._devices = tuple(_module_device(model) for model in models)
        cuda_devices = tuple(
            device for device in self._devices if device.type == "cuda"
        )
        if len(set(cuda_devices)) != len(cuda_devices):
            raise ValueError(
                "Expected at most one model replica per CUDA device"
            )

        self._call_lock = RLock()
        self._state: _FittedState | None = None
        self.eval()
        self._models_object_id = id(self.models)
        self._model_signature = _model_signature(self.models)
        self._workers = tuple(
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"sdm-ensemble-{worker_id}",
            )
            for worker_id in range(len(models))
        )

    @property
    def rng_policy(self) -> Literal["canonical", "member"]:
        r"""The immutable model-core random stream policy."""
        return self._rng_policy

    @property
    def member_plan(self) -> EnsembleMemberPlan:
        r"""Placement-independent fitted estimator RNG plan evidence."""
        self._validate_model_authority()
        if self._state is None:
            raise RuntimeError(
                "'member_plan' requires a successful call to 'fit'"
            )
        return self._state.member_plan

    @property
    def member_plan_digest(self) -> str:
        r"""Digest of the fitted logical RNG member plan.

        Worker placement is intentionally excluded, so equal one-worker and
        multi-worker executions have the same digest.
        """
        return self.member_plan.digest_sha256

    @_serialized
    @torch.inference_mode()
    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        callbacks: Sequence[Callback] | None = None,
        **kwargs: Any,
    ) -> TableTensor:
        r"""Run one-shot ensemble inference across the replicas."""
        self._validate_execution(generator=generator)
        callbacks = () if callbacks is None else callbacks
        self._validate_callbacks(callbacks)
        for callback in callbacks:
            callback.on_forward_start(
                self,
                x_context,
                y_context,
                x_query,
                related_context_tables,
                related_query_tables,
                recipe=recipe,
                num_estimators=num_estimators,
                generator=generator,
                callbacks=callbacks,
                **kwargs,
            )

        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(y_context, TableTensor):
            y_context = TableTensor.from_tensor(y_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        output_device = x_query.device
        kwargs = self._model(0)._context_kwargs(x_context, kwargs)

        if (related_context_tables is None) != (related_query_tables is None):
            raise ValueError(
                "Expected 'related_context_tables' and "
                "'related_query_tables' to be provided together"
            )
        if related_query_tables is not None:
            assert related_context_tables is not None
            related_query_tables = related_query_tables.select_tables(
                tables=related_context_tables.tables
            )

        execution = RecipeExecution(
            self._model(0).default_recipe()
            if recipe is None
            else copy.deepcopy(recipe)
        )
        with torch.amp.autocast(x_query.device.type, enabled=False):
            contexts = execution.fit_transform(
                x=x_context,
                y=y_context,
                related_tables=related_context_tables,
                num_members=num_estimators,
                generator=generator,
            )
            queries = execution.transform(
                x=x_query,
                related_tables=related_query_tables,
            )

        queries = tuple(
            self._apply_callbacks(query, callbacks) for query in queries
        )
        plan = _member_plan(
            self._rng_policy,
            generator,
            num_estimators,
        )
        jobs = self._member_jobs(num_estimators)
        futures = [
            self._workers[worker_id].submit(
                self._forward_worker,
                worker_id,
                tuple(
                    (member_id, contexts[member_id], queries[member_id])
                    for member_id in member_ids
                ),
                tuple(plan.seeds[member_id] for member_id in member_ids),
                plan.shared_generator,
                kwargs,
                output_device,
                _autocast_state(self._devices[worker_id]),
            )
            for worker_id, member_ids in jobs
        ]
        results = _future_results(futures)
        outputs = _restore_outputs(results, num_estimators, output_device)
        prediction = self._finalize(
            execution=execution,
            outputs=outputs,
            is_regression=contexts[0].y.numerical.size(-1) > 0,
            device=output_device,
        )

        for callback in callbacks:
            callback.on_forward_end(self, prediction)
        return prediction

    @_serialized
    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Fit member-local caches and keep them on assigned devices."""
        self._validate_execution(generator=generator)
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)
        kwargs = self._model(0)._context_kwargs(x, kwargs)

        self.clear()
        execution = RecipeExecution(
            self._model(0).default_recipe()
            if recipe is None
            else copy.deepcopy(recipe)
        )
        with torch.amp.autocast(x.device.type, enabled=False):
            contexts = execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                generator=generator,
            )

        plan = _member_plan(
            self._rng_policy,
            generator,
            num_estimators,
        )
        jobs = self._member_jobs(num_estimators)
        futures = [
            self._workers[worker_id].submit(
                self._fit_worker,
                worker_id,
                tuple((member_id, contexts[member_id]) for member_id in ids),
                tuple(plan.seeds[member_id] for member_id in ids),
                plan.shared_generator,
                kwargs,
                _autocast_state(self._devices[worker_id]),
            )
            for worker_id, ids in jobs
        ]
        results = _future_results(futures)
        members: list[_MemberState | None] = [None] * num_estimators
        for worker_results in results:
            for member_id, worker_id, cache in worker_results:
                members[member_id] = _MemberState(worker_id, cache)
        if any(member is None for member in members):
            raise RuntimeError("Expected one fitted cache per estimator")

        self._validate_model_authority()
        self._state = _FittedState(
            execution=execution,
            members=tuple(cast(_MemberState, member) for member in members),
            kwargs=kwargs,
            is_regression=contexts[0].y.numerical.size(-1) > 0,
            member_plan=plan.evidence,
        )

    @_serialized
    @torch.inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        r"""Predict with fitted member-local caches in parallel."""
        self._validate_execution(generator=None, require_generator=False)
        callbacks = () if callbacks is None else callbacks
        self._validate_callbacks(callbacks)
        for callback in callbacks:
            callback.on_forward_start(
                self,
                x,
                related_tables,
                callbacks=callbacks,
            )

        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        output_device = x.device
        if self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )
        state = self._state

        first_cache = state.members[0].cache
        if related_tables is not None:
            schema = cast(
                RelatedTablesSchema | None,
                first_cache["related_tables_schema"],
            )
            if schema is None:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(tables=schema.tables)

        with torch.amp.autocast(x.device.type, enabled=False):
            queries = state.execution.transform(x, related_tables)
        queries = tuple(
            self._apply_callbacks(query, callbacks) for query in queries
        )

        jobs: dict[int, list[int]] = {}
        for member_id, member in enumerate(state.members):
            jobs.setdefault(member.worker_id, []).append(member_id)
        futures = [
            self._workers[worker_id].submit(
                self._predict_worker,
                worker_id,
                tuple(
                    (
                        member_id,
                        queries[member_id],
                        state.members[member_id].cache,
                    )
                    for member_id in member_ids
                ),
                state.kwargs,
                output_device,
                _autocast_state(self._devices[worker_id]),
            )
            for worker_id, member_ids in jobs.items()
        ]
        results = _future_results(futures)
        outputs = _restore_outputs(
            results,
            len(state.members),
            output_device,
        )
        prediction = self._finalize(
            execution=state.execution,
            outputs=outputs,
            is_regression=state.is_regression,
            device=output_device,
        )

        for callback in callbacks:
            callback.on_forward_end(self, prediction)
        return prediction

    @_serialized
    def clear(self) -> None:
        r"""Clear all fitted member caches."""
        self._state = None

    @_serialized
    def close(self) -> None:
        r"""Wait for work to finish and close the worker threads."""
        self.clear()
        for worker in self._workers:
            worker.shutdown(wait=True)

    def __getstate__(self) -> dict[str, object]:
        raise RuntimeError(
            "EnsembleParallel cannot be serialized while its workers are live"
        )

    def _member_jobs(
        self,
        num_estimators: int,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        jobs = tuple(
            tuple(range(worker_id, num_estimators, len(self.models)))
            for worker_id in range(min(len(self.models), num_estimators))
        )
        return tuple(
            (worker_id, member_ids)
            for worker_id, member_ids in enumerate(jobs)
        )

    def _model(self, worker_id: int) -> ICLModel:
        return cast(ICLModel, self.models[worker_id])

    @torch.inference_mode()
    def _fit_worker(
        self,
        worker_id: int,
        members: tuple[tuple[int, MemberContext], ...],
        seeds: tuple[int | None, ...],
        shared_generator: torch.Generator | None,
        kwargs: dict[str, Any],
        autocast: _AutocastState,
    ) -> list[tuple[int, int, Cache]]:
        model = self._model(worker_id)
        device = self._devices[worker_id]
        results = []
        with torch.autocast(
            device.type,
            enabled=autocast.enabled,
            dtype=autocast.dtype,
        ):
            for (member_id, context), seed in zip(members, seeds, strict=True):
                try:
                    context = _move_context(context, device)
                    model._validate_context(
                        x=context.x,
                        y=context.y,
                        related_tables=context.related_tables,
                    )
                    cache = _member_cache(context)
                    model._forward(
                        x_context=context.x,
                        y_context=context.y,
                        x_query=None,
                        related_context_tables=context.related_tables,
                        related_query_tables=None,
                        cache=cache,
                        generator=_member_generator(
                            device,
                            seed,
                            shared_generator,
                        ),
                        **kwargs,
                    )
                    results.append((member_id, worker_id, cache.freeze()))
                except Exception as error:  # noqa: BLE001
                    _raise_worker_error(
                        error,
                        stage="fit",
                        member_id=member_id,
                        worker_id=worker_id,
                        device=device,
                    )
        return results

    @torch.inference_mode()
    def _forward_worker(
        self,
        worker_id: int,
        members: tuple[tuple[int, MemberContext, MemberQuery], ...],
        seeds: tuple[int | None, ...],
        shared_generator: torch.Generator | None,
        kwargs: dict[str, Any],
        output_device: torch.device,
        autocast: _AutocastState,
    ) -> list[_WorkerOutput]:
        model = self._model(worker_id)
        device = self._devices[worker_id]
        results = []
        with torch.autocast(
            device.type,
            enabled=autocast.enabled,
            dtype=autocast.dtype,
        ):
            for (member_id, context, query), seed in zip(
                members, seeds, strict=True
            ):
                try:
                    context = _move_context(context, device)
                    query = _move_query(query, device)
                    model._validate_context(
                        x=context.x,
                        y=context.y,
                        related_tables=context.related_tables,
                    )
                    model._validate_query(
                        x_context=context.x.schema,
                        x_query=query.x,
                        related_context_tables=context.related_tables.schema
                        if context.related_tables is not None
                        else None,
                        related_query_tables=query.related_tables,
                    )
                    output = model._forward(
                        x_context=context.x,
                        y_context=context.y,
                        x_query=query.x,
                        related_context_tables=context.related_tables,
                        related_query_tables=query.related_tables,
                        cache=None,
                        generator=_member_generator(
                            device,
                            seed,
                            shared_generator,
                        ),
                        **kwargs,
                    )
                    output = cast(TableTensor, output.to(query.x.dtype))
                    results.append(
                        _worker_output(
                            member_id,
                            output,
                            output_device,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    _raise_worker_error(
                        error,
                        stage="forward",
                        member_id=member_id,
                        worker_id=worker_id,
                        device=device,
                    )
        return results

    @torch.inference_mode()
    def _predict_worker(
        self,
        worker_id: int,
        members: tuple[tuple[int, MemberQuery, Cache], ...],
        kwargs: dict[str, Any],
        output_device: torch.device,
        autocast: _AutocastState,
    ) -> list[_WorkerOutput]:
        model = self._model(worker_id)
        device = self._devices[worker_id]
        results = []
        with torch.autocast(
            device.type,
            enabled=autocast.enabled,
            dtype=autocast.dtype,
        ):
            for member_id, query, cache in members:
                try:
                    query = _move_query(query, device)
                    model._validate_query(
                        x_context=cast(TableSchema, cache["x_schema"]),
                        x_query=query.x,
                        related_context_tables=cast(
                            RelatedTablesSchema | None,
                            cache["related_tables_schema"],
                        ),
                        related_query_tables=query.related_tables,
                    )
                    output = model._forward(
                        x_context=None,
                        y_context=None,
                        x_query=query.x,
                        related_context_tables=None,
                        related_query_tables=query.related_tables,
                        cache=cache,
                        generator=None,
                        **kwargs,
                    )
                    output = cast(TableTensor, output.to(query.x.dtype))
                    results.append(
                        _worker_output(
                            member_id,
                            output,
                            output_device,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    _raise_worker_error(
                        error,
                        stage="predict",
                        member_id=member_id,
                        worker_id=worker_id,
                        device=device,
                    )
        return results

    def _apply_callbacks(
        self,
        query: MemberQuery,
        callbacks: Sequence[Callback],
    ) -> MemberQuery:
        x = query.x
        related_tables = query.related_tables
        for callback in callbacks:
            x, related_tables = callback.on_preprocessing_end(
                self,
                x,
                related_tables,
            )
        return MemberQuery(x=x, related_tables=related_tables)

    @staticmethod
    def _validate_callbacks(callbacks: Sequence[Callback]) -> None:
        if any(callback.requires_grad for callback in callbacks):
            raise ValueError(
                "EnsembleParallel does not support callbacks requiring "
                "gradients"
            )

    def _validate_execution(
        self,
        *,
        generator: torch.Generator | None,
        require_generator: bool = True,
    ) -> None:
        self._validate_model_authority()
        num_workers = len(self._workers)
        if (
            len(self.models) != num_workers
            or len(self._devices) != num_workers
        ):
            raise RuntimeError("EnsembleParallel worker topology was modified")
        if self._rng_policy == "canonical":
            if num_workers != 1:
                raise RuntimeError(
                    "The 'canonical' RNG policy requires exactly one model "
                    "replica"
                )
        elif self._rng_policy == "member":
            if require_generator and generator is None:
                raise ValueError(
                    "The 'member' RNG policy requires an explicit generator"
                )
        else:
            raise AssertionError(
                f"Unsupported RNG policy {self._rng_policy!r}"
            )
        if (
            self._state is not None
            and self._state.member_plan.policy != self._rng_policy
        ):
            raise RuntimeError(
                "Current RNG policy does not match the fitted member plan"
            )

    def _validate_model_authority(self) -> None:
        r"""Fail closed if an explicit replica changed after construction."""
        try:
            unchanged = (
                not self.training
                and id(self.models) == self._models_object_id
                and _model_signature(self.models) == self._model_signature
            )
        except Exception as error:
            raise _model_authority_error() from error
        if not unchanged:
            raise _model_authority_error()

    @staticmethod
    def _finalize(
        execution: RecipeExecution,
        outputs: tuple[TableTensor, ...],
        is_regression: bool,
        device: torch.device,
    ) -> TableTensor:
        if is_regression:
            with torch.amp.autocast(device.type, enabled=False):
                outputs = execution.inverse_transform_target(outputs)
        with torch.amp.autocast(device.type, enabled=False):
            return execution.transform_output(outputs)


def _model_authority_error() -> RuntimeError:
    return RuntimeError(
        "EnsembleParallel model topology or state changed after construction; "
        "close it and construct a new runner"
    )


def _model_signature(
    models: Iterable[torch.nn.Module],
) -> tuple[_ModelSignature, ...]:
    signatures = []
    for model in models:
        parameters = tuple(
            _tensor_signature("parameter", name, tensor)
            for name, tensor in model.named_parameters(remove_duplicate=False)
        )
        buffers = tuple(
            _tensor_signature("buffer", name, tensor)
            for name, tensor in model.named_buffers(remove_duplicate=False)
        )
        signatures.append(
            _ModelSignature(
                object_id=id(model),
                model_type=type(model),
                training=model.training,
                device=_module_device(model),
                tensors=parameters + buffers,
            )
        )
    return tuple(signatures)


def _tensor_signature(
    kind: Literal["parameter", "buffer"],
    name: str,
    tensor: Tensor,
) -> _TensorSignature:
    return _TensorSignature(
        kind=kind,
        name=name,
        object_id=id(tensor),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
        data_ptr=tensor.data_ptr(),
        version=tensor._version,
        requires_grad=tensor.requires_grad,
    )


def _module_device(model: torch.nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if len(devices) == 0:
        return torch.device("cpu")
    if len(devices) != 1:
        raise ValueError("Expected each model replica to occupy one device")
    return next(iter(devices))


def _member_cache(context: MemberContext) -> Cache:
    return Cache(
        x_schema=context.x.schema,
        related_tables_schema=context.related_tables.schema
        if context.related_tables is not None
        else None,
        classes=(
            context.y.categorical.categories[0]
            if context.y.categorical.size(-1) > 0
            else None
        ),
    )


def _move_context(
    context: MemberContext,
    device: torch.device,
) -> MemberContext:
    return MemberContext(
        x=cast(TableTensor, context.x.to(device)),
        y=cast(TableTensor, context.y.to(device)),
        related_tables=context.related_tables.to(device)
        if context.related_tables is not None
        else None,
    )


def _move_query(query: MemberQuery, device: torch.device) -> MemberQuery:
    return MemberQuery(
        x=cast(TableTensor, query.x.to(device)),
        related_tables=query.related_tables.to(device)
        if query.related_tables is not None
        else None,
    )


def _member_plan(
    policy: Literal["canonical", "member"],
    generator: torch.Generator | None,
    num_members: int,
) -> _MemberPlan:
    state_before = _generator_state_digest(generator)
    if policy == "canonical":
        evidence = _plan_evidence(
            policy=policy,
            seeds=(None,) * num_members,
            state_before=state_before,
            state_after=state_before,
        )
        return _MemberPlan(
            seeds=(None,) * num_members,
            shared_generator=generator,
            evidence=evidence,
        )

    assert policy == "member"
    if generator is None:
        raise ValueError(
            "The 'member' RNG policy requires an explicit generator"
        )
    seeds = torch.randint(
        0,
        2**63 - 1,
        (num_members,),
        generator=generator,
        device=generator.device,
        dtype=torch.int64,
    )
    seed_values = tuple(seeds.cpu().tolist())
    evidence = _plan_evidence(
        policy=policy,
        seeds=seed_values,
        state_before=state_before,
        state_after=_generator_state_digest(generator),
    )
    return _MemberPlan(
        seeds=seed_values,
        shared_generator=None,
        evidence=evidence,
    )


def _member_generator(
    device: torch.device,
    seed: int | None,
    shared_generator: torch.Generator | None,
) -> torch.Generator | None:
    if shared_generator is not None:
        return shared_generator
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def _generator_state_digest(generator: torch.Generator | None) -> str | None:
    if generator is None:
        return None
    state = generator.get_state().cpu()
    return hashlib.sha256(bytes(state.tolist())).hexdigest()


def _plan_evidence(
    *,
    policy: Literal["canonical", "member"],
    seeds: tuple[int | None, ...],
    state_before: str | None,
    state_after: str | None,
) -> EnsembleMemberPlan:
    payload: dict[str, object] = {
        "version": 1,
        "policy": policy,
        "member_seeds": list(seeds),
        "generator_state_before_sha256": state_before,
        "generator_state_after_sha256": state_after,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return EnsembleMemberPlan(
        version=1,
        policy=policy,
        member_seeds=seeds,
        generator_state_before_sha256=state_before,
        generator_state_after_sha256=state_after,
        digest_sha256=digest,
    )


def _autocast_state(device: torch.device) -> _AutocastState:
    return _AutocastState(
        enabled=torch.is_autocast_enabled(device.type),
        dtype=torch.get_autocast_dtype(device.type),
    )


def _future_results(futures: Sequence[Future[T]]) -> list[T]:
    wait(futures)
    return [future.result() for future in futures]


def _worker_output(
    member_id: int,
    output: TableTensor,
    output_device: torch.device,
) -> _WorkerOutput:
    output = cast(TableTensor, output.to(output_device))
    event = None
    if output_device.type == "cuda":
        with torch.cuda.device(output_device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(output_device))
    return _WorkerOutput(member_id, output, event)


def _restore_outputs(
    results: Sequence[Sequence[_WorkerOutput]],
    num_members: int,
    output_device: torch.device,
) -> tuple[TableTensor, ...]:
    outputs: list[TableTensor | None] = [None] * num_members
    for worker_results in results:
        for result in worker_results:
            if result.event is not None:
                torch.cuda.current_stream(output_device).wait_event(
                    result.event
                )
            outputs[result.member_id] = result.output
    if any(output is None for output in outputs):
        raise RuntimeError("Expected one output per estimator")
    return tuple(cast(TableTensor, output) for output in outputs)


def _raise_worker_error(
    error: Exception,
    *,
    stage: str,
    member_id: int,
    worker_id: int,
    device: torch.device,
) -> NoReturn:
    if device.type == "cuda":
        with suppress(Exception):
            torch.cuda.current_stream(device).synchronize()
    raise RuntimeError(
        f"Ensemble member {member_id} failed during {stage} on worker "
        f"{worker_id} ({device})"
    ) from error
