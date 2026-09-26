# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import copy
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal, cast

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
from sdm.cache import Cache
from sdm.models.callback import Callback
from sdm.processing.execution import (
    MemberContext,
    MemberQuery,
    RecipeExecution,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema


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

    #: Cells per estimator batch with ``estimator_batch_size="auto"``, as
    #: counted by :meth:`_estimator_cells`. Larger batches add memory without
    #: speeding up inference on GPUs.
    _estimator_batch_cells: ClassVar[int] = 2**20

    #: Cells each table row adds for per-row work such as in-context learning.
    _estimator_row_cells: ClassVar[int] = 32

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
        estimator_batch_size: int | Literal["auto"] | None = "auto",
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
            estimator_batch_size: Maximum number of consecutive estimators run
                through the model in one call. ``"auto"`` (default) batches
                estimators up to a size budget for their preprocessed tables,
                and runs them one by one when gradients are required. ``1``
                runs estimators one by one, which minimizes device memory;
                ``None`` batches as many as possible. Estimators whose
                preprocessed tables differ in shape, category counts or
                classes, or that come with related tables, run in separate
                calls. Batched and sequential predictions are equal up to
                floating-point rounding.
            callbacks: Callbacks applied in sequence to this model call. The
                preprocessing hooks of all estimators run before their model
                forward hooks.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self.training
        requires_grad |= any(callback.requires_grad for callback in callbacks)

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
            contexts=contexts,
            queries=queries,
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
        estimator_batch_size: int | Literal["auto"] | None = "auto",
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
            estimator_batch_size: Maximum number of consecutive estimators run
                through the model in one call. ``"auto"`` (default) batches
                estimators up to a size budget for their preprocessed tables.
                ``1`` runs estimators one by one, which minimizes device
                memory; ``None`` batches as many as possible. Estimators whose
                preprocessed tables differ in shape, category counts or
                classes, or that come with related tables, run in separate
                calls. Estimators fitted together are predicted together; with
                ``"auto"`` and without callbacks, :meth:`predict` splits their
                query rows into chunks that keep the budget. Batched and
                sequential predictions are equal up to floating-point rounding.
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

        members = [
            self._prepare_context(context, callbacks) for context in contexts
        ]
        values = _member_class_values(members, estimator_batch_size)
        batches = _plan_batches(
            contexts=members,
            queries=None,
            class_values=values,
            estimator_batch_size=estimator_batch_size,
            max_cells=self._estimator_batch_cells,
            cells=self._estimator_cells,
        )
        cache = Cache(
            recipe_execution=recipe_execution,
            kwargs=kwargs,
            num_batches=len(batches),
            max_cells=(
                self._estimator_batch_cells
                if estimator_batch_size == "auto"
                else None
            ),
        )
        for i, batch in enumerate(batches):
            with inference_mode("no_grad"):
                class_values = _class_values(values[batch])
                context = _stack_context(members[batch])
                categorical_mask = _categorical_mask(members[batch])
                batch_cache = Cache(
                    x_schemas=tuple(
                        member.x.schema for member in members[batch]
                    ),
                    y_schema=context.y.schema,
                    related_tables_schema=context.related_tables.schema
                    if context.related_tables is not None
                    else None,
                    classes=(
                        context.y.categorical.categories[0]
                        if context.y.categorical.size(-1) > 0
                        else None
                    ),
                    class_values=class_values,
                    categorical_mask=categorical_mask,
                )
                self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=None,
                    related_context_tables=context.related_tables,
                    related_query_tables=None,
                    cache=batch_cache,
                    generator=generator,
                    categorical_mask=categorical_mask,
                    num_members=batch.stop - batch.start,
                    **kwargs,
                )

            if x.is_cuda and len(contexts) > 1:
                try:  # Copy to pinned CPU memory:
                    batch_cache = batch_cache._apply_tensor(
                        lambda tensor: torch.ops.aten._to_copy.default(
                            tensor,
                            device="cpu",
                            pin_memory=True,
                            non_blocking=True,  # Required for `pin_memory`.
                        )
                    )
                finally:
                    torch.cuda.current_stream(x.device).synchronize()

            cache[i] = batch_cache

        self._cache = cache.freeze()

    def predict(
        self,
        x: Tensor | TableTensor | EnsembleTable,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""Predict unseen query examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.
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
        requires_grad = any(callback.requires_grad for callback in callbacks)

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
        num_batches = cast(int, self._cache["num_batches"])
        # Callbacks see every estimator output once, so queries stay whole.
        max_cells = None if callbacks else self._cache["max_cells"]
        caches = [cast(Cache, self._cache[i]) for i in range(num_batches)]
        next_cache = caches[0]

        compute_stream: torch.cuda.Stream | None = None
        transfer_stream: torch.cuda.Stream | None = None
        try:
            if x.is_cuda:
                compute_stream = torch.cuda.current_stream(x.device)
                if x.device not in self._transfer_streams:
                    transfer_stream = torch.cuda.Stream(x.device)
                    self._transfer_streams[x.device] = transfer_stream
                else:
                    transfer_stream = self._transfer_streams[x.device]
                with torch.cuda.stream(transfer_stream):
                    next_cache = next_cache.to(x.device, non_blocking=True)

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
            for i in range(len(caches)):
                cache, next_cache = next_cache, None
                assert cache is not None

                x_schemas = cast(tuple[TableSchema, ...], cache["x_schemas"])
                classes = cast(Tensor | None, cache["classes"])
                members = [
                    self._prepare_query(
                        query=query,
                        x_schema=x_schema,
                        related_tables_schema=cast(
                            RelatedTablesSchema | None,
                            cache["related_tables_schema"],
                        ),
                        callbacks=callbacks,
                    )
                    for query, x_schema in zip(
                        queries[start : start + len(x_schemas)],
                        x_schemas,
                        strict=True,
                    )
                ]
                start += len(x_schemas)

                if i + 1 < len(caches):
                    next_cache = caches[i + 1]
                if x.is_cuda and next_cache is not None:
                    assert transfer_stream is not None
                    with torch.cuda.stream(transfer_stream):
                        next_cache = next_cache.to(x.device, non_blocking=True)

                chunks = [
                    self._forward_batch(
                        contexts=None,
                        queries=chunk,
                        cache=cache,
                        categorical_mask=cast(
                            Tensor, cache["categorical_mask"]
                        ),
                        class_values=cast(
                            tuple[tuple[Any, ...], ...] | None,
                            cache["class_values"],
                        ),
                        callbacks=callbacks,
                        requires_grad=requires_grad,
                        generator=None,
                        **cast(dict[str, Any], self._cache["kwargs"]),
                    )
                    for chunk in _query_chunks(
                        queries=members,
                        max_cells=cast(int | None, max_cells),
                        cells=self._estimator_cells,
                        num_classes=(
                            0 if classes is None else classes.numel()
                        ),
                    )
                ]
                outs += (
                    chunks[0]
                    if len(chunks) == 1
                    else [
                        cast(TableTensor, torch.cat(parts, dim=-2))
                        for parts in zip(*chunks, strict=True)
                    ]
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
        categorical_mask: Tensor,
        num_members: int = 1,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        r"""Run the model on preprocessed tables of one estimator batch.

        Tables carry shape ``[E, ..., R, D]`` when ``E > 1`` estimators run
        together and ``[..., R, D]`` otherwise. Column names and category
        vocabularies are those of the first estimator; per-estimator column
        and class order is not observable from the tables.

        Args:
            x_context: The feature tensor of in-context examples, or ``None``
                when replaying a cache.
            y_context: The targets of in-context examples, or ``None`` when
                replaying a cache.
            x_query: The feature tensor of query examples, or ``None`` when
                recording a cache.
            related_context_tables: Related context for in-context examples.
            related_query_tables: Related context for query examples.
            cache: The cache to record into or replay from, or ``None``.
                Recorded tensors are replayed with queries of the same
                estimator batch.
            generator: Pseudorandom number generator for model execution.
            categorical_mask: Boolean ``[C]`` or ``[E, 1, ..., C]`` tensor
                broadcastable to ``x.numerical.size()[:-2] + (C,)`` marking
                numerical feature columns that were categorical before
                preprocessing. Passed on recording, replaying and uncached
                calls alike.
            num_members: The number of estimators ``E`` in the batch.
                Model-side randomness must be drawn per estimator, in order,
                so that a batch matches separate calls.
            kwargs: Additional keyword arguments passed by the caller.

        Returns:
            Predictions of shape ``[..., R_query, *]``. Classification outputs
            hold exactly one column per class in the order of
            ``y_context.categorical.categories[0]`` (``cache["classes"]`` on
            replay), each named by its class value; :class:`ICLModel` relabels
            them per estimator when stacked.
        """

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _estimator_cells(self, x: TableTensor, num_classes: int) -> int:
        r"""Count one estimator's table against ``_estimator_batch_cells``.

        Rows of the preprocessed table ``x`` count their columns plus
        ``_estimator_row_cells``. ``num_classes`` is ``0`` for regression.
        Models that run an estimator as several tasks scale the count up.
        """
        rows = math.prod(x.size()[:-1])
        return rows * (x.size(-1) + self._estimator_row_cells)

    def _forward_members(
        self,
        contexts: Sequence[MemberContext],
        queries: Sequence[MemberQuery],
        *,
        estimator_batch_size: int | Literal["auto"] | None = "auto",
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> list[TableTensor]:
        r"""Run recipe-transformed members on the device of their queries.

        Context features and targets may live on another device; each batch
        is moved when it runs.
        Returns one output per member before target inversion and
        ``recipe.output``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = self.training
        requires_grad |= any(callback.requires_grad for callback in callbacks)
        if requires_grad and estimator_batch_size == "auto":
            estimator_batch_size = 1

        members = [
            self._prepare_context(context, callbacks) for context in contexts
        ]
        queries = [
            self._prepare_query(
                query=query,
                x_schema=member.x.schema,
                related_tables_schema=member.related_tables.schema
                if member.related_tables is not None
                else None,
                callbacks=callbacks,
            )
            for member, query in zip(members, queries, strict=True)
        ]
        values = _member_class_values(members, estimator_batch_size)
        outs: list[TableTensor] = []
        for batch in _plan_batches(
            contexts=members,
            queries=queries,
            class_values=values,
            estimator_batch_size=estimator_batch_size,
            max_cells=self._estimator_batch_cells,
            cells=self._estimator_cells,
        ):
            device = queries[batch.start].x.device
            outs += self._forward_batch(
                contexts=[
                    member._replace(
                        x=cast(TableTensor, member.x.to(device)),
                        y=cast(TableTensor, member.y.to(device)),
                    )
                    for member in members[batch]
                ],
                queries=queries[batch],
                cache=None,
                categorical_mask=None,
                class_values=_class_values(values[batch]),
                callbacks=callbacks,
                requires_grad=requires_grad,
                generator=generator,
                **kwargs,
            )
        return outs

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

    def _forward_batch(
        self,
        contexts: Sequence[MemberContext] | None,
        queries: Sequence[MemberQuery],
        *,
        cache: Cache | None,
        categorical_mask: Tensor | None,
        class_values: tuple[tuple[Any, ...], ...] | None,
        callbacks: Sequence[Callback],
        requires_grad: bool,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> list[TableTensor]:
        # Stacking inside the autograd region keeps callback-captured leaves
        # attached to the graph.
        with inference_mode("grad" if requires_grad else "inference"):
            context = None if contexts is None else _stack_context(contexts)
            if categorical_mask is None:
                assert contexts is not None
                categorical_mask = _categorical_mask(contexts)
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
                num_members=len(queries),
                **kwargs,
            )
            outs = _unstack(out, class_values, len(queries))
            for i in range(len(outs)):
                for callback in callbacks:
                    outs[i] = callback.on_model_forward_end(self, outs[i])
        return [cast(TableTensor, out.to(query.x.dtype)) for out in outs]

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
    # torch.stack aligns columns by name, which would undo per-estimator column
    # shuffles; renaming to the first member's names stacks blocks by position.
    columns = cast(Mapping[StypeLike, Sequence[str]], ref.columns)
    renamed: list[Tensor] = [
        ref,
        *(
            table.__class__(columns=columns, **dict(table.items()))
            for table in tables[1:]
        ),
    ]
    return cast(TableTensor, torch.stack(renamed))


def _stack_context(members: Sequence[MemberContext]) -> MemberContext:
    if len(members) == 1:
        return members[0]
    return MemberContext(
        x=_stack([member.x for member in members]),
        y=_stack([member.y for member in members]),
        related_tables=None,
        input_stypes=members[0].input_stypes,
    )


def _stack_query(members: Sequence[MemberQuery]) -> MemberQuery:
    if len(members) == 1:
        return members[0]
    return MemberQuery(
        x=_stack([member.x for member in members]), related_tables=None
    )


def _plan_batches(
    contexts: Sequence[MemberContext],
    queries: Sequence[MemberQuery] | None,
    class_values: Sequence[tuple[Any, ...] | None],
    estimator_batch_size: int | Literal["auto"] | None,
    max_cells: int,
    cells: Callable[[TableTensor, int], int],
) -> list[slice]:
    # Runs of consecutive members whose tables stack, cut at the batch size or,
    # for "auto", before the cell budget is exceeded. Keeping members in order
    # keeps model-side randomness in the order of separate calls.
    batches: list[slice] = []
    start = total = 0
    key: Hashable = None
    for i, context in enumerate(contexts):
        query = None if queries is None else queries[i]
        member_key = _stack_key(context, query, class_values[i])
        num_classes = _num_classes(context)
        member_cells = cells(context.x, num_classes)
        if query is not None:
            member_cells += cells(query.x, num_classes)
        if i > start and (
            member_key is None
            or member_key != key
            or i - start == estimator_batch_size
            or (
                estimator_batch_size == "auto"
                and total + member_cells > max_cells
            )
        ):
            batches.append(slice(start, i))
            start, total = i, 0
        key = member_key
        total += member_cells
    batches.append(slice(start, len(contexts)))
    return batches


def _stack_key(
    context: MemberContext,
    query: MemberQuery | None,
    class_values: tuple[Any, ...] | None,
) -> Hashable:
    # Members stack when their tables agree in block shapes, dtypes and
    # category counts and their targets in classes; ``None`` never stacks.
    if context.related_tables is not None or (
        query is not None and query.related_tables is not None
    ):
        return None
    tables = (
        [context.x, context.y]
        if query is None
        else [context.x, context.y, query.x]
    )
    return (
        tuple(
            (
                tuple(
                    (stype, block.size(), block.dtype)
                    for stype, block in table.items()
                ),
                tuple(c.numel() for c in table.categorical.categories),
            )
            for table in tables
        ),
        None if class_values is None else frozenset(class_values),
    )


def _num_classes(member: MemberContext) -> int:
    y = member.y.categorical
    return y.categories[0].numel() if y.size(-1) > 0 else 0


def _query_chunks(
    queries: Sequence[MemberQuery],
    max_cells: int | None,
    cells: Callable[[TableTensor, int], int],
    num_classes: int,
) -> list[list[MemberQuery]]:
    # Row chunks of a batch's queries within the cell budget, but never smaller
    # than the queries of one estimator on their own.
    total = sum(cells(query.x, num_classes) for query in queries)
    if max_cells is None or len(queries) == 1 or total <= max_cells:
        return [list(queries)]
    rows = queries[0].x.size(-2)
    budget = max(max_cells, total // len(queries))
    splits = [
        query.x.split(max(1, budget * rows // total), dim=-2)
        for query in queries
    ]
    return [
        [MemberQuery(x=x, related_tables=None) for x in xs]
        for xs in zip(*splits, strict=True)
    ]


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
        return mask[0]
    # Insert the member's batch dimensions so the mask broadcasts over them:
    return mask.view(len(members), *(1,) * (x.dim() - 2), -1)  # [E, 1, ..., C]


def _member_class_values(
    members: Sequence[MemberContext],
    estimator_batch_size: int | Literal["auto"] | None,
) -> list[tuple[Any, ...] | None]:
    # Classes of every member, read only when members may share a batch.
    if estimator_batch_size == 1 or members[0].y.categorical.size(-1) == 0:
        return [None] * len(members)
    return [
        tuple(member.y.categorical.categories[0].tolist())
        for member in members
    ]


def _class_values(
    values: Sequence[tuple[Any, ...] | None],
) -> tuple[tuple[Any, ...], ...] | None:
    if len(values) == 1 or values[0] is None:
        return None
    return cast(tuple[tuple[Any, ...], ...], tuple(values))


def _unstack(
    out: TableTensor,
    class_values: tuple[tuple[Any, ...], ...] | None,
    num_members: int,
) -> list[TableTensor]:
    outs = (
        [out]
        if num_members == 1
        else list(cast(tuple[TableTensor, ...], out.unbind(0)))
    )
    if class_values is None:
        return outs
    # The model labels columns in the first member's class order; member `e`'s
    # column `j` holds class `class_values[e][j]`.
    labels = out.columns[Stype.numerical]
    index = {value: i for i, value in enumerate(class_values[0])}
    return [
        TableTensor(
            columns={Stype.numerical: [labels[index[v]] for v in values]},
            numerical=member_out.numerical,
        )
        for member_out, values in zip(outs, class_values, strict=True)
    ]
