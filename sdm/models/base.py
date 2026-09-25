# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import (
    EnsembleTable,
    Recipe,
    RelatedTables,
    Stype,
    StypeLike,
    TableTensor,
    Task,
    TaskLike,
)
from sdm._inference import inference_mode
from sdm._warnings import warn_once
from sdm.cache import Cache, KVCacheEntry
from sdm.models.callback import Callback
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema

_INCOMPATIBLE_ESTIMATORS = (
    "Estimators in one batch must share column names and classes; use "
    "'estimator_batch_size=1'"
)


class ICLModel(torch.nn.Module, abc.ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`ICLModel` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
    """

    #: Semantic types supported for input columns in this model.
    supported_feature_stypes: ClassVar[frozenset[Stype]]

    #: Semantic types supported for target columns in this model.
    supported_target_stypes: ClassVar[frozenset[Stype]]

    #: Prediction tasks supported in this model.
    supported_tasks: ClassVar[frozenset[Task]]

    #: Whether this model supports multi-target predictions.
    supports_multi_target: ClassVar[bool]

    #: Whether this model supports additional related context.
    supports_related_tables: ClassVar[bool]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "supported_target_stypes"):
            cls.supported_tasks = frozenset(
                Task.from_stype(stype) for stype in cls.supported_target_stypes
            )

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
    ) -> None:
        super().__init__()

        if task is None:
            self.tasks = self.supported_tasks
        elif isinstance(task, str):
            self.tasks = frozenset({Task(task)})
        else:
            self.tasks = frozenset({Task(t) for t in task})

        if not self.tasks.issubset(self.supported_tasks):
            invalid = ", ".join(
                f"{str(task)!r}" for task in self.tasks - self.supported_tasks
            )
            raise ValueError(
                f"{self.__class__.__name__!r} received unsupported tasks "
                f"{invalid}"
            )

        self._cache: Cache | None = None
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}

    def forward(
        self,
        x_context: Tensor | TableTensor | EnsembleTable,  # [..., R_context, D]
        y_context: Tensor | TableTensor | EnsembleTable,  # [..., R_context, 1]
        x_query: Tensor | TableTensor | EnsembleTable,  # [..., R_query, D]
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        estimator_batch_size: int | None = 1,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""The in-context learning forward pass.

        Args:
            x_context: The feature tensor of in-context examples with shape
                ``[..., R_context, D]`` with ``R_context`` rows and ``D``
                columns.
            y_context: The targets of in-context examples with shape
                ``[..., R_context, 1]``.
            x_query: The feature tensor of query examples with shape
                ``[..., R_query, D]`` with ``R_query`` rows and ``D`` columns.
            related_context_tables: Related context for in-context examples.
            related_query_tables: Related context for query examples.
            recipe: The custom recipe for pre- and post-processing.
            num_estimators: The number of estimators ``E`` for ensembling.
                If ``None``, the leading dimension of higher-rank inputs is
                used as the estimator dimension, allowing input data to be
                customized per estimator (*e.g.*, different in-context examples
                per estimator).
            estimator_batch_size: Maximum number of estimators run through the
                model in one call. ``1`` (default) runs estimators one by one,
                exactly as before; ``None`` runs all of them together.
                Estimators in one batch must share column names, table shapes
                and class counts, and related tables require ``1``; device
                memory grows with the batch size. Model-side randomness drawn
                per call (*e.g.*, the error-correcting output codes of
                :class:`~sdm.models.KumoTabular` for more than 10 classes) is
                shared within a batch, so such predictions differ from the
                sequential ones.
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self._requires_grad(callbacks)

        if (related_context_tables is None) != (related_query_tables is None):
            raise ValueError(
                "Expected 'related_context_tables' and 'related_query_tables' "
                "to be provided together"
            )

        if related_query_tables is not None:
            assert related_context_tables is not None
            related_query_tables = related_query_tables.select_tables(
                tables=related_context_tables.tables
            )

        recipe_execution = RecipeExecution(
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        with (
            torch.amp.autocast(x_query.device.type, enabled=False),
            inference_mode("no_grad" if requires_grad else "inference"),
        ):
            contexts = recipe_execution.fit_transform(
                x=x_context,
                y=y_context,
                related_tables=related_context_tables,
                num_members=num_estimators,
                generator=generator,
            )
        with (
            torch.amp.autocast(x_query.device.type, enabled=False),
            inference_mode("no_grad" if requires_grad else "inference"),
        ):
            queries = recipe_execution.transform(
                x=x_query,
                related_tables=related_query_tables,
            )

        outs = self._forward_members(
            contexts,
            queries,
            estimator_batch_size=estimator_batch_size,
            callbacks=callbacks,
            generator=generator,
            **kwargs,
        )

        # Regression: invert target before stacking estimator outputs.
        if contexts[0].y.numerical.size(-1) > 0:
            with (
                torch.amp.autocast(x_query.device.type, enabled=False),
                inference_mode("grad" if requires_grad else "inference"),
            ):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with (
            torch.amp.autocast(x_query.device.type, enabled=False),
            inference_mode("grad" if requires_grad else "inference"),
        ):
            return recipe_execution.transform_output(outs)

    def fit(
        self,
        x: Tensor | TableTensor | EnsembleTable,  # [..., R, D]
        y: Tensor | TableTensor | EnsembleTable,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Fit and cache in-context examples.

        Repeated calls to :meth:`predict` can then reuse the same in-context
        examples while only providing new query examples.

        Args:
            x: The feature tensor of in-context examples with shape
                ``[..., R, D]`` with ``R`` rows and ``C`` columns.
            y: The targets of in-context examples with shape
                ``[..., R, 1]``.
            related_tables: Related context for in-context examples.
            recipe: The custom recipe for pre- and post-processing.
            num_estimators: The number of estimators ``E`` for ensembling.
                If ``None``, the leading dimension of higher-rank inputs is
                used as the estimator dimension, allowing input data to be
                customized per estimator (*e.g.*, different in-context examples
                per estimator).
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.
        """
        callbacks = () if callbacks is None else callbacks

        self.clear()

        recipe_execution = RecipeExecution(
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode("no_grad"),
        ):
            contexts = recipe_execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                generator=generator,
            )

        cache = Cache(
            recipe_execution=recipe_execution,
            kwargs=kwargs,
        )
        for i, context in enumerate(contexts):
            context = self._prepare_context(context, callbacks)
            # Tensors recorded by the model live in `state` so that caches of
            # several estimators can be stacked for batched prediction.
            state = Cache(categorical_mask=_categorical_mask([context]))
            estimator_cache = Cache(
                x_schema=context.x.schema,
                y_schema=context.y.schema,
                related_tables_schema=context.related_tables.schema
                if context.related_tables is not None
                else None,
                classes=(
                    context.y.categorical.categories[0]
                    if context.y.categorical.size(-1) > 0
                    else None
                ),
                state=state,
            )

            with inference_mode("no_grad"):
                self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=None,
                    related_context_tables=context.related_tables,
                    related_query_tables=None,
                    cache=state,
                    generator=generator,
                    categorical_mask=cast(Tensor, state["categorical_mask"]),
                    **kwargs,
                )

            if x.is_cuda and len(contexts) > 1:
                try:  # Copy to pinned CPU memory:
                    estimator_cache = estimator_cache._apply_tensor(
                        lambda tensor: torch.ops.aten._to_copy.default(
                            tensor,
                            device="cpu",
                            pin_memory=True,
                            non_blocking=True,  # Required for `pin_memory`.
                        )
                    )
                finally:
                    torch.cuda.current_stream(x.device).synchronize()

            cache[i] = estimator_cache

        self._cache = cache.freeze()

    def predict(
        self,
        x: Tensor | TableTensor | EnsembleTable,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        estimator_batch_size: int | None = 1,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""Predict unseen query examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.
            estimator_batch_size: Maximum number of estimators run through the
                model in one call. ``1`` (default) runs estimators one by one,
                exactly as before; ``None`` runs all of them together.
                Estimators in one batch must share column names, table shapes
                and class counts, and related tables require ``1``; device
                memory grows with the batch size. Independent of how the model
                was fitted. Models whose cache holds non-tensor state (*e.g.*,
                more than 10 classes in :class:`~sdm.models.TabICLv2` or
                :class:`~sdm.models.KumoTabular`) require ``1``.
            callbacks: Callbacks applied in sequence to this model call.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
        if self.training:
            raise RuntimeError(
                f"{self.__class__.__name__!r}.predict() does not support "
                "gradient-based training through a fitted context cache. "
                "To fix, call `model.eval()`."
            )

        callbacks = () if callbacks is None else callbacks
        requires_grad = self._requires_grad(callbacks)

        if self._cache is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        if related_tables is not None:
            if cast(Cache, self._cache[0])["related_tables_schema"] is None:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(
                tables=cast(
                    RelatedTablesSchema,
                    cast(Cache, self._cache[0])["related_tables_schema"],
                ).tables,
            )

        recipe_execution = cast(
            RecipeExecution,
            self._cache["recipe_execution"],
        )
        caches = [
            cast(Cache, self._cache[i])
            for i in range(recipe_execution.num_members)
        ]
        if estimator_batch_size is None:
            estimator_batch_size = len(caches)
        size = min(estimator_batch_size, len(caches))
        batches = [
            caches[start : start + size]
            for start in range(0, len(caches), size)
        ]
        if size > 1:
            self._check_batchable(batches)

        compute_stream: torch.cuda.Stream | None = None
        transfer_stream: torch.cuda.Stream | None = None

        def _fetch(batch: Sequence[Cache]) -> Cache:
            # Assemble the replay cache of one estimator batch on the device.
            states = [cast(Cache, cache["state"]) for cache in batch]
            with torch.cuda.stream(transfer_stream):
                state = (
                    states[0]
                    if len(states) == 1
                    else Cache.stack(
                        states,
                        device=x.device,
                        non_blocking=True,
                    )
                )
                replay = Cache(
                    state,
                    classes=batch[0]["classes"],
                    y_schema=batch[0]["y_schema"],
                )
                return replay.to(x.device, non_blocking=True).freeze()

        try:
            if x.is_cuda:
                compute_stream = torch.cuda.current_stream(x.device)
                if x.device not in self._transfer_streams:
                    transfer_stream = torch.cuda.Stream(x.device)
                    self._transfer_streams[x.device] = transfer_stream
                else:
                    transfer_stream = self._transfer_streams[x.device]
            next_cache = _fetch(batches[0])

            with (
                torch.amp.autocast(x.device.type, enabled=False),
                inference_mode("no_grad" if requires_grad else "inference"),
            ):
                queries = recipe_execution.transform(x, related_tables)

            if x.is_cuda:
                assert compute_stream is not None
                assert transfer_stream is not None
                compute_stream.wait_stream(transfer_stream)

            outs: list[TableTensor] = []
            start = 0
            for i, batch in enumerate(batches):
                cache, next_cache = next_cache, None
                assert cache is not None
                members = [
                    self._prepare_query(
                        query,
                        cast(TableSchema, estimator_cache["x_schema"]),
                        cast(
                            RelatedTablesSchema | None,
                            estimator_cache["related_tables_schema"],
                        ),
                        callbacks,
                    )
                    for estimator_cache, query in zip(
                        batch,
                        queries[start : start + len(batch)],
                        strict=True,
                    )
                ]
                start += len(batch)

                if i + 1 < len(batches):
                    next_cache = _fetch(batches[i + 1])

                categorical_mask = cast(Tensor, cache["categorical_mask"])
                if len(batch) > 1:
                    categorical_mask = categorical_mask.view(
                        len(batch),
                        *(1,) * (members[0].x.dim() - 2),
                        categorical_mask.size(-1),
                    )  # [E, 1, ..., C]
                # The per-estimator class tensors stay on the host whenever
                # more than one estimator is cached.
                classes = None
                if cache["classes"] is not None:
                    classes = [
                        cast(Tensor, estimator_cache["classes"])
                        for estimator_cache in batch
                    ]
                outs += self._forward_batch(
                    None,
                    members,
                    cache=cache,
                    categorical_mask=categorical_mask,
                    classes=classes,
                    callbacks=callbacks,
                    requires_grad=requires_grad,
                    generator=None,
                    **cast(dict[str, Any], self._cache["kwargs"]),
                )

                if x.is_cuda:
                    assert compute_stream is not None
                    for tensor in cache._tensors():
                        tensor.record_stream(compute_stream)

                if x.is_cuda and next_cache is not None:
                    assert compute_stream is not None
                    assert transfer_stream is not None
                    compute_stream.wait_stream(transfer_stream)

        except BaseException:
            if transfer_stream is not None:
                transfer_stream.synchronize()
            raise

        # Regression: invert target before stacking estimator outputs.
        if cast(Cache, self._cache[0])["classes"] is None:
            with (
                torch.amp.autocast(x.device.type, enabled=False),
                inference_mode("grad" if requires_grad else "inference"),
            ):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode("grad" if requires_grad else "inference"),
        ):
            return recipe_execution.transform_output(outs)

    def clear(self) -> None:
        r"""Clear cached context state created by :meth:`fit`."""
        self._cache = None

    def __getstate__(self) -> dict[str, object]:
        for stream in self._transfer_streams.values():
            stream.synchronize()
        state = super().__getstate__()
        state.pop("_transfer_streams", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._transfer_streams = {}

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"

    # Abstract Methods ########################################################

    @abc.abstractmethod
    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, Y]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,  # [C] or [E, 1, ..., C]
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        r"""Run the model on preprocessed tables.

        Tables carry a leading estimator dimension ``[E, ..., R, D]`` when
        ``E > 1`` estimators run together and ``[..., R, D]`` otherwise. Column
        names and category vocabularies are those of the first estimator; the
        per-estimator column and class order is not observable from the
        tables. ``categorical_mask`` broadcasts to
        ``x.numerical.size()[:-2] + (C,)`` and marks numerical feature columns
        that were categorical before preprocessing.

        Tensors written to ``cache`` keep the leading batch dimensions of the
        inputs first, so caches of separately fitted estimators can be stacked
        for batched prediction. On replay, ``cache["classes"]`` and
        ``cache["y_schema"]`` describe the fitted target.

        Classification outputs hold one column per class in the order of
        ``y_context.categorical.categories[0]`` (``cache["classes"]`` on
        replay), named by class value.
        """

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _forward_members(
        self,
        contexts: Sequence[MemberContext],
        queries: Sequence[MemberQuery],
        *,
        estimator_batch_size: int | None = 1,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> list[TableTensor]:
        r"""Run recipe-transformed members that live on the model device.

        Returns one output per member before target inversion and
        ``recipe.output``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self._requires_grad(callbacks)
        if estimator_batch_size is None:
            estimator_batch_size = len(contexts)
        outs: list[TableTensor] = []
        for start in range(0, len(contexts), estimator_batch_size):
            members = [
                self._prepare_context(context, callbacks)
                for context in contexts[start : start + estimator_batch_size]
            ]
            query_members = [
                self._prepare_query(
                    query,
                    member.x.schema,
                    member.related_tables.schema
                    if member.related_tables is not None
                    else None,
                    callbacks,
                )
                for member, query in zip(
                    members,
                    queries[start : start + estimator_batch_size],
                    strict=True,
                )
            ]
            outs += self._forward_batch(
                members,
                query_members,
                cache=None,
                categorical_mask=_categorical_mask(members),
                classes=_classes(members),
                callbacks=callbacks,
                requires_grad=requires_grad,
                generator=generator,
                **kwargs,
            )
        return outs

    def _forward_batch(
        self,
        contexts: Sequence[MemberContext] | None,
        queries: Sequence[MemberQuery],
        *,
        cache: Cache | None,
        categorical_mask: Tensor,
        classes: Sequence[Tensor] | None,
        callbacks: Sequence[Callback],
        requires_grad: bool,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> list[TableTensor]:
        # Stack inside the autograd region so that leaves captured by
        # callbacks stay attached to the graph.
        with inference_mode("grad" if requires_grad else "inference"):
            context = None if contexts is None else _stack_context(contexts)
            query = _stack_query(queries)
            out = self._forward(
                x_context=None if context is None else context.x,
                y_context=None if context is None else context.y,
                x_query=query.x,
                related_context_tables=(
                    None if context is None else context.related_tables
                ),
                related_query_tables=query.related_tables,
                cache=cache,
                generator=generator,
                categorical_mask=categorical_mask,
                **kwargs,
            )
            outs = _unstack(out, classes, len(queries))
            for i, member_out in enumerate(outs):
                for callback in callbacks:
                    member_out = callback.on_model_forward_end(
                        self, member_out
                    )
                outs[i] = member_out

        return [cast(TableTensor, out.to(query.x.dtype)) for out in outs]

    def _prepare_context(
        self,
        context: MemberContext,
        callbacks: Sequence[Callback],
    ) -> MemberContext:
        for callback in callbacks:
            x, y, related_tables = callback.on_context_preprocessing_end(
                self,
                context.x,
                context.y,
                context.related_tables,
            )
            context = context._replace(x=x, y=y, related_tables=related_tables)
        self._validate_context(
            x=context.x,
            y=context.y,
            related_tables=context.related_tables,
        )
        return context

    def _prepare_query(
        self,
        query: MemberQuery,
        x_schema: TableSchema,
        related_tables_schema: RelatedTablesSchema | None,
        callbacks: Sequence[Callback],
    ) -> MemberQuery:
        for callback in callbacks:
            query = MemberQuery(
                *callback.on_query_preprocessing_end(self, *query)
            )
        self._validate_query(
            x_context=x_schema,
            x_query=query.x,
            related_context_tables=related_tables_schema,
            related_query_tables=query.related_tables,
        )
        return query

    def _requires_grad(self, callbacks: Sequence[Callback]) -> bool:
        return self.training or any(
            callback.requires_grad for callback in callbacks
        )

    def _check_batchable(self, batches: Sequence[Sequence[Cache]]) -> None:
        ref = batches[0][0]
        if ref["related_tables_schema"] is not None:
            raise ValueError("Related tables require 'estimator_batch_size=1'")
        if not all(
            isinstance(value, Tensor | KVCacheEntry)
            for value in cast(Cache, ref["state"]).values()
        ):
            raise ValueError(
                f"{self.__class__.__name__!r} cache cannot be batched across "
                f"estimators; use 'estimator_batch_size=1'"
            )
        for batch in batches:
            layouts = {
                (
                    _column_sets(cast(TableSchema, cache["x_schema"]).columns),
                    _num_classes(cache),
                )
                for cache in batch
            }
            if len(layouts) > 1:
                raise ValueError(_INCOMPATIBLE_ESTIMATORS)

    def _validate_context(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> None:

        if not self.supports_multi_target and y.size(-1) != 1:
            raise ValueError(
                f"Expected target to have exactly one column "
                f"(got {y.size(-1)})"
            )
        if x.size()[:-1] != y.size()[:-1]:
            raise ValueError(
                f"Expected features and targets to have matching row "
                f"dimensions (got {tuple(x.size()[:-1])} and "
                f"{tuple(y.size()[:-1])})"
            )
        invalid = x.active_stypes - self.supported_feature_stypes - {Stype.id}
        if len(invalid) > 0:
            stypes = ", ".join(f"{str(stype)!r}" for stype in invalid)
            warn_once(
                key="model-unsupported-feature-stypes",
                message=(
                    f"{self.__class__.__name__!r} received unsupported "
                    f"feature stypes {stypes}. Columns with unsupported "
                    f"feature stypes will not be consumed by the model."
                ),
            )
        invalid = y.active_stypes - self.supported_target_stypes
        if len(invalid) > 0:
            stypes = ", ".join(f"{str(stype)!r}" for stype in invalid)
            raise ValueError(
                f"{self.__class__.__name__!r} received unsupported target "
                f"stypes {stypes}"
            )
        invalid = y.active_stypes - {task.stype for task in self.tasks}
        if len(invalid) > 0:
            tasks = ", ".join(
                f"{str(Task.from_stype(stype))!r}" for stype in invalid
            )
            raise ValueError(
                f"{self.__class__.__name__!r} is not initialized for tasks "
                f"{tasks}"
            )

        if related_tables is not None:
            if not self.supports_related_tables:
                raise ValueError(
                    f"{self.__class__.__name__!r} does not support related "
                    f"tables"
                )
            for table_name, table in related_tables.tables.items():
                invalid = table.active_stypes - self.supported_feature_stypes
                invalid = invalid - {Stype.id}
                if len(invalid) > 0:
                    stypes = ", ".join(f"{str(stype)!r}" for stype in invalid)
                    warn_once(
                        key="model-unsupported-feature-stypes",
                        message=(
                            f"{self.__class__.__name__!r} received "
                            f"unsupported feature stypes {stypes} in related "
                            f"table {table_name!r}. Columns with unsupported "
                            f"feature stypes will not be consumed by the "
                            f"model."
                        ),
                    )

    def _validate_query(
        self,
        x_context: TableSchema,
        x_query: TableTensor,
        related_context_tables: RelatedTablesSchema | None,
        related_query_tables: RelatedTables[TableTensor] | None,
    ) -> None:

        if x_context != x_query.schema:
            raise ValueError(
                "Expected context and query features to share the same schema"
            )

        if (related_context_tables is None) != (related_query_tables is None):
            raise ValueError("Expected related tables to be provided together")

        if related_context_tables is not None:
            assert related_query_tables is not None
            if not related_query_tables.schema.is_subset_of(
                related_context_tables
            ):
                raise ValueError(
                    "Expected related context and query tables to share the "
                    "same schema"
                )


def _stack(tables: Sequence[TableTensor]) -> TableTensor:
    ref = tables[0]
    if len(tables) == 1:
        return ref
    # `torch.stack` aligns columns by name, which would undo per-estimator
    # column shuffles. Renaming to the first member's names stacks by position.
    renamed: list[Tensor] = [
        ref,
        *(
            table.__class__(
                columns=cast(Mapping[StypeLike, Sequence[str]], ref.columns),
                **dict(table.items()),
            )
            for table in tables[1:]
        ),
    ]
    return cast(TableTensor, torch.stack(renamed))


def _column_sets(
    columns: Mapping[Stype, tuple[str, ...]],
) -> frozenset[tuple[Stype, frozenset[str]]]:
    return frozenset(
        (stype, frozenset(names)) for stype, names in columns.items()
    )


def _num_classes(cache: Cache) -> int | None:
    classes = cast(Tensor | None, cache["classes"])
    return None if classes is None else classes.numel()


def _check_compatible(tables: Sequence[TableTensor]) -> None:
    layouts = {
        (
            _column_sets(table.columns),
            tuple(c.numel() for c in table.categorical.categories),
        )
        for table in tables
    }
    if len(layouts) > 1:
        raise ValueError(_INCOMPATIBLE_ESTIMATORS)


def _stack_context(members: Sequence[MemberContext]) -> MemberContext:
    if len(members) == 1:
        return members[0]
    if members[0].related_tables is not None:
        raise ValueError("Related tables require 'estimator_batch_size=1'")
    _check_compatible([member.x for member in members])
    _check_compatible([member.y for member in members])
    return MemberContext(
        x=_stack([member.x for member in members]),
        y=_stack([member.y for member in members]),
        related_tables=None,
        input_stypes=members[0].input_stypes,
    )


def _stack_query(members: Sequence[MemberQuery]) -> MemberQuery:
    if len(members) == 1:
        return members[0]
    _check_compatible([member.x for member in members])
    return MemberQuery(
        x=_stack([member.x for member in members]),
        related_tables=None,
    )


def _categorical_mask(members: Sequence[MemberContext]) -> Tensor:
    x = members[0].x
    mask = torch.tensor(
        [
            [
                member.input_stypes.get(column) == Stype.categorical
                for column in member.x.columns[Stype.numerical]
            ]
            for member in members
        ],
        dtype=torch.bool,
        device=x.device,
    )  # [E, C]
    if len(members) == 1:
        return mask[0]  # [C]
    return mask.view(
        len(members),
        *(1,) * (x.dim() - 2),
        mask.size(-1),
    )  # [E, 1, ..., C]


def _classes(members: Sequence[MemberContext]) -> list[Tensor] | None:
    if members[0].y.categorical.size(-1) == 0:
        return None
    return [member.y.categorical.categories[0] for member in members]


def _member_columns(
    labels: Sequence[str],
    classes: Sequence[Tensor],
) -> tuple[tuple[str, ...], ...]:
    # The model labels its columns in the first member's class order; member
    # `e` holds class `values[e][j]` in column `j`.
    values = [member_classes.tolist() for member_classes in classes]
    index = {value: i for i, value in enumerate(values[0])}
    if any(value not in index for member in values[1:] for value in member):
        raise ValueError(_INCOMPATIBLE_ESTIMATORS)
    return tuple(
        tuple(labels[index[value]] for value in member) for member in values
    )


def _unstack(
    out: TableTensor,
    classes: Sequence[Tensor] | None,
    num_members: int,
) -> list[TableTensor]:
    if num_members == 1:
        return [out]
    outs = list(cast(tuple[TableTensor, ...], out.unbind(0)))
    labels = out.columns[Stype.numerical]
    if classes is None or len(labels) != classes[0].numel():
        return outs
    return [
        TableTensor(
            columns={Stype.numerical: names},
            numerical=member_out.numerical,
        )
        for member_out, names in zip(
            outs,
            _member_columns(labels, classes),
            strict=True,
        )
    ]
