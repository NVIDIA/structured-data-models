# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import copy
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

import sdm.processing as sp
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

# Feature processors allowed together with `seqused_train`/`seqused_cols`.
# The counts identify padding purely by position, so only processors known
# to keep every column in place qualify; matched by exact type since a
# subclass may change the layout.
_SEQUSED_LAYOUT_PRESERVING = (sp.Identity, sp.Clip)


def _iter_processor_leaves(
    processor: torch.nn.Module,
) -> Iterator[torch.nn.Module]:
    # The two containers only delegate to what they wrap; everything else
    # (including dispatchers and container subclasses) is judged as a leaf.
    if type(processor) is sp.Sequential:
        for child in processor:
            yield from _iter_processor_leaves(child)
    elif type(processor) is sp.EnsembleProcessorAdapter:
        yield from _iter_processor_leaves(processor.processor)
    else:
        yield processor


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
        seqused_train: Tensor | None = None,  # [...]
        seqused_cols: Tensor | None = None,  # []
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
                through the model in one call. ``1`` (default) runs estimators
                one by one; ``None`` batches as many as possible. Estimators
                whose preprocessed tables differ in shape or target class
                set, or that come with related tables, run in separate
                calls. Device memory grows with the batch size.
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            seqused_train: Valid in-context example counts with shape
                ``[...]`` and :external+torch:ref:`torch.int32 <dtype-doc>`
                dtype.
                When set, only the first ``seqused_train`` of the
                ``R_context`` in-context rows act as context; the remaining
                rows are padding, masked from every attention key/value
                stream so they cannot influence any prediction. Padded
                feature entries must be finite and of moderate magnitude:
                pad with ``0`` or with copies of real rows, since large
                magnitudes overflow the padded rows' internal statistics
                to ``NaN``, which masking does not remove (beyond about
                ``1e19`` in float32/bfloat16, already between ``1e4`` and
                ``1e5`` under float16 autocast, whose largest finite value
                is 65504). The same limit applies to padded
                regression targets. Padded ``y_context`` entries must
                still be valid targets: for categorical targets they must
                repeat class values that already occur among the valid
                rows, since the class set is derived from the full padded
                column and any other padded value silently widens it.
                Padding with ``0`` is only safe for regression targets (or
                when ``0`` is a valid in-context class).
                Out-of-range counts are clamped and produce degenerate
                predictions rather than errors.
                Together with padded query rows (whose extra outputs callers
                discard), this lets streams of varying table sizes be padded
                to a small set of bucketed shapes so compiled graphs and
                per-shape kernel selection are reused across tables.
                Requires a pass-through ``recipe`` since fitted
                pre-processing would derive its state from the padded rows;
                only element-wise, column-layout-preserving pre-processing
                is supported, as stateless processors that change the
                column layout (such as ``SelectColumns``) desynchronize the
                counts from the columns they mask.
                Padded classification is limited to the model's native
                class count (10 for :class:`~sdm.models.TabICLv2`); tables
                with more classes take the hierarchical classification
                route and raise a ``ValueError`` when combined with
                ``seqused_train``.
            seqused_cols: Valid column count as a scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, only the first ``seqused_cols`` columns act as
                features; the remaining columns are padding, excluded from
                feature grouping and masked from row-wise attention. The
                count refers to the numerical block handed to the model
                after recipe feature processing
                (``x_context.numerical.size(-1)`` under a pass-through
                ``recipe``), not the table width - non-numerical columns
                are removed before masking applies.
                The same finite, moderate-magnitude contract as
                ``seqused_train`` applies to padded columns; out-of-range
                counts are clamped and produce
                degenerate predictions rather than errors.
                Unlike ``seqused_train``, the count is shared across batch
                elements.
                Both padding keywords index batch elements of a shared
                context, so they cannot be combined with ensemble-aware
                inputs: :class:`~sdm.EnsembleTable` inputs are
                rejected, and higher-rank inputs require an explicit
                ``num_estimators`` so their leading dimension keeps its
                batch interpretation instead of being consumed as the
                estimator dimension.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        # Only forward the padding keywords when set so that subclasses
        # implementing a narrower private hook keep working. Merging here
        # lets the per-estimator `self._forward(**kwargs)` calls receive the
        # counts alongside any other keyword arguments.
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols

        # Argument validation runs up front so invalid padding keywords
        # fail before any pre-processing work.
        self._validate_seqused(seqused_train, seqused_cols)
        self._validate_seqused_inputs(
            num_estimators=num_estimators,
            seqused_train=seqused_train,
            seqused_cols=seqused_cols,
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
        )
        self._warn_seqused_cols_table(
            x=x_context,
            seqused_cols=seqused_cols,
            param_name="x_context",
        )
        self._validate_seqused_recipe(recipe, seqused_train, seqused_cols)

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
        estimator_batch_size: int | None = 1,
        callbacks: Sequence[Callback] | None = None,
        generator: torch.Generator | None = None,
        seqused_train: Tensor | None = None,  # [...]
        seqused_cols: Tensor | None = None,  # []
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
                through the model in one call. ``1`` (default) runs estimators
                one by one; ``None`` batches as many as possible. Estimators
                whose preprocessed tables differ in shape or target class
                set, or that come with related tables, run in separate
                calls. Device memory grows with the batch size. Estimators
                fitted together are predicted together.
            callbacks: Callbacks applied in sequence to this model call.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            seqused_train: Valid in-context example counts with shape
                ``[...]`` and :external+torch:ref:`torch.int32 <dtype-doc>`
                dtype.
                When set, rows beyond the per-element count are padding and
                are masked from the cached key/value projections; subsequent
                :meth:`predict` calls reuse the count automatically.
                Classification tables with more classes than the model's
                native head (10 for :class:`~sdm.models.TabICLv2`) raise a
                ``ValueError`` when combined with ``seqused_train``; the
                class count is only known once the target has been
                pre-processed, so the previous fit has been cleared by
                then. See :meth:`forward` for the padding contract.
            seqused_cols: Valid column count as a scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, columns beyond the count are padding; subsequent
                :meth:`predict` calls reuse the count and must pass ``x``
                padded to the same number of columns. See :meth:`forward`
                for the padding contract.
            kwargs: Additional keyword arguments passed to the model. They
                are cached and re-applied by :meth:`predict`.
        """
        callbacks = () if callbacks is None else callbacks

        self._validate_seqused(seqused_train, seqused_cols)
        self._validate_seqused_inputs(
            num_estimators=num_estimators,
            seqused_train=seqused_train,
            seqused_cols=seqused_cols,
            x=x,
            y=y,
        )
        self._warn_seqused_cols_table(x, seqused_cols, param_name="x")
        self._validate_seqused_recipe(recipe, seqused_train, seqused_cols)

        # Only forward the padding keywords when set so that subclasses
        # implementing a narrower private hook keep working. The counts
        # join the cached keyword arguments that :meth:`predict` replays;
        # cloning detaches them from the caller's buffers.
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train.clone()
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols.clone()

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

        contexts = [
            self._prepare_context(
                context,
                callbacks,
                seqused="seqused_train" in kwargs or "seqused_cols" in kwargs,
            )
            for context in contexts
        ]
        class_values = _class_values(contexts, estimator_batch_size)
        batches = _batch_slices(
            contexts=contexts,
            queries=None,
            class_values=class_values,
            estimator_batch_size=estimator_batch_size,
        )
        cache = Cache(
            recipe_execution=recipe_execution,
            kwargs=kwargs,
            num_batches=len(batches),
        )
        for i, batch in enumerate(batches):
            with inference_mode("no_grad"):
                context = _stack_context(contexts[batch])
                categorical_mask = _categorical_mask(contexts[batch])
                batch_cache = Cache(
                    x_schemas=tuple(
                        context.x.schema for context in contexts[batch]
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
                    class_values=class_values[batch],
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
                If :meth:`fit` was called with ``seqused_train`` or
                ``seqused_cols``, the cached counts are replayed so padded
                in-context rows stay masked; with ``seqused_cols``, ``x``
                must be padded to the same number of columns as the fitted
                rows.
            related_tables: Related context for query examples.
            callbacks: Callbacks applied in sequence to this model call.
                When :meth:`fit` cached ``seqused_cols``, callbacks that
                return a replacement query raise :class:`ValueError`, since
                the count also identifies padded query columns by position.

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
        caches = [cast(Cache, self._cache[i]) for i in range(num_batches)]
        next_cache = caches[0]
        # The cached keyword arguments live outside the per-estimator caches
        # moved below, so the counts cloned by `fit` are moved here.
        kwargs = dict(cast(dict[str, Any], self._cache["kwargs"]))
        for key in ("seqused_train", "seqused_cols"):
            if key in kwargs:
                kwargs[key] = cast(Tensor, kwargs[key]).to(x.device)
        seqused_cols = "seqused_cols" in kwargs

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
                batch_queries = [
                    self._prepare_query(
                        query=query,
                        x_schema=x_schema,
                        related_tables_schema=cast(
                            RelatedTablesSchema | None,
                            cache["related_tables_schema"],
                        ),
                        callbacks=callbacks,
                        seqused_cols=seqused_cols,
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

                outs += self._forward_batch(
                    contexts=None,
                    queries=batch_queries,
                    cache=cache,
                    categorical_mask=cast(Tensor, cache["categorical_mask"]),
                    class_values=cast(
                        Sequence[tuple[Any, ...] | None] | None,
                        cache["class_values"],
                    ),
                    callbacks=callbacks,
                    requires_grad=requires_grad,
                    generator=None,
                    **kwargs,
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
        requires_grad = self.training
        requires_grad |= any(callback.requires_grad for callback in callbacks)

        if estimator_batch_size == 1 and len(contexts) > 1:
            outs: list[TableTensor] = []
            for context, query in zip(contexts, queries, strict=True):
                outs.extend(
                    self._forward_members(
                        contexts=(context,),
                        queries=(query,),
                        estimator_batch_size=1,
                        callbacks=callbacks,
                        generator=generator,
                        **kwargs,
                    )
                )
            return outs

        contexts = [
            self._prepare_context(
                context,
                callbacks,
                seqused="seqused_train" in kwargs or "seqused_cols" in kwargs,
            )
            for context in contexts
        ]
        queries = [
            self._prepare_query(
                query=query,
                x_schema=context.x.schema,
                related_tables_schema=context.related_tables.schema
                if context.related_tables is not None
                else None,
                callbacks=callbacks,
                seqused_cols="seqused_cols" in kwargs,
            )
            for context, query in zip(contexts, queries, strict=True)
        ]
        class_values = _class_values(contexts, estimator_batch_size)
        outs: list[TableTensor] = []
        for batch in _batch_slices(
            contexts=contexts,
            queries=queries,
            class_values=class_values,
            estimator_batch_size=estimator_batch_size,
        ):
            outs += self._forward_batch(
                contexts=contexts[batch],
                queries=queries[batch],
                cache=None,
                categorical_mask=None,
                class_values=class_values[batch],
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
        *,
        seqused: bool = False,
    ) -> MemberContext:
        for callback in callbacks:
            x, y, related_tables = callback.on_context_preprocessing_end(
                self,
                context.x,
                context.y,
                context.related_tables,
            )
            if seqused and (
                x is not context.x
                or y is not context.y
                or related_tables is not context.related_tables
            ):
                raise ValueError(
                    "Callbacks cannot return a replacement context with "
                    "seqused padding; return the context unchanged to "
                    "preserve the padded trailing rows and columns"
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
        *,
        seqused_cols: bool = False,
    ) -> MemberQuery:
        for callback in callbacks:
            x, related_tables = callback.on_query_preprocessing_end(
                self, *query
            )
            if seqused_cols and (
                x is not query.x or related_tables is not query.related_tables
            ):
                raise ValueError(
                    "Callbacks cannot return a replacement query with "
                    "seqused_cols; return the query unchanged to "
                    "preserve the padded trailing columns"
                )
            query = MemberQuery(x=x, related_tables=related_tables)
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
        class_values: Sequence[tuple[Any, ...] | None] | None,
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

    def _validate_seqused(
        self,
        seqused_train: Tensor | None,
        seqused_cols: Tensor | None,
    ) -> None:
        # Only metadata is checked; reading the counts would sync the device
        # on every call. Out-of-range counts are clamped where consumed.
        if seqused_train is not None and seqused_train.dtype != torch.int32:
            raise ValueError(
                f"'seqused_train' must have dtype torch.int32 "
                f"(got {seqused_train.dtype})"
            )
        if seqused_cols is not None:
            if seqused_cols.dtype != torch.int32:
                raise ValueError(
                    f"'seqused_cols' must have dtype torch.int32 "
                    f"(got {seqused_cols.dtype})"
                )
            if seqused_cols.dim() != 0:
                raise ValueError(
                    f"'seqused_cols' must be a scalar tensor "
                    f"(got shape {tuple(seqused_cols.size())})"
                )

    def _validate_seqused_inputs(
        self,
        num_estimators: int | None,
        seqused_train: Tensor | None,
        seqused_cols: Tensor | None,
        **tables: Tensor | TableTensor | EnsembleTable | None,
    ) -> None:
        if seqused_train is None and seqused_cols is None:
            return

        # The counts index batch elements of one shared context, so they
        # cannot be aligned with per-estimator tables.
        for name, table in tables.items():
            if table is None:
                continue
            if isinstance(table, EnsembleTable):
                raise ValueError(
                    f"'seqused_train'/'seqused_cols' cannot be combined "
                    f"with the ensemble-aware 'EnsembleTable' input "
                    f"{name!r}: the padding counts describe one shared "
                    f"context and cannot be aligned with per-estimator "
                    f"tables"
                )
            # With 'num_estimators=None' the leading dimension would be
            # consumed as the estimator dimension instead of the batch.
            if num_estimators is None and table.dim() > 2:
                raise ValueError(
                    f"'seqused_train'/'seqused_cols' with the higher-rank "
                    f"input {name!r} is ambiguous when 'num_estimators' is "
                    f"None, since the leading dimension would be consumed "
                    f"as the estimator dimension while the padding counts "
                    f"index batch elements; pass 'num_estimators' "
                    f"explicitly (e.g., 'num_estimators=1') to keep the "
                    f"batch interpretation"
                )
            # One count per batch element, i.e. the dimensions ahead of the
            # trailing row and column dimensions.
            if (
                seqused_train is not None
                and seqused_train.size() != table.size()[:-2]
            ):
                raise ValueError(
                    f"'seqused_train' must match the batch dimensions "
                    f"{tuple(table.size()[:-2])} of {name!r} "
                    f"(got shape {tuple(seqused_train.size())})"
                )

    def _validate_seqused_recipe(
        self,
        recipe: Recipe | None,
        seqused_train: Tensor | None,
        seqused_cols: Tensor | None,
    ) -> None:
        if seqused_train is None and seqused_cols is None:
            return

        # A fitted recipe derives its state from the padded rows as well.
        effective = self.default_recipe() if recipe is None else recipe
        if effective.features.requires_fit or effective.target.requires_fit:
            raise ValueError(
                "Recipe pre-processing fits its state on the padded rows "
                "and targets, letting padding influence predictions in "
                "violation of the seqused contract; pass a pass-through "
                "recipe such as 'sdm.Recipe()' together with "
                "'seqused_train'/'seqused_cols'"
            )

        # A stateless processor may still select, reorder, or synthesize
        # columns, moving the padded trailing columns away from the mask.
        for leaf in _iter_processor_leaves(effective.features):
            if type(leaf) not in _SEQUSED_LAYOUT_PRESERVING:
                raise ValueError(
                    f"Recipe feature processing contains "
                    f"{leaf.__class__.__name__!r}, which cannot be verified "
                    f"to preserve the column layout, so the padded trailing "
                    f"rows and columns counted by "
                    f"'seqused_train'/'seqused_cols' may no longer identify "
                    f"the padding after pre-processing; pass a "
                    f"layout-preserving recipe such as 'sdm.Recipe()' "
                    f"together with 'seqused_train'/'seqused_cols'"
                )

    def _warn_seqused_cols_table(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        seqused_cols: Tensor | None,
        *,
        param_name: str,
    ) -> None:
        if (
            seqused_cols is not None
            and isinstance(x, TableTensor)
            and x.size(-1) != x.numerical.size(-1)
        ):
            warn_once(
                key="model-seqused-cols-table-columns",
                message=(
                    f"'seqused_cols' counts columns of the extracted "
                    f"numerical block ({x.numerical.size(-1)} column(s)), "
                    f"but {param_name!r} has {x.size(-1)} table column(s); "
                    f"non-numerical columns (including id data) are removed "
                    f"before masking applies."
                ),
                stacklevel=3,
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


def _batch_slices(
    contexts: Sequence[MemberContext],
    queries: Sequence[MemberQuery] | None,
    class_values: Sequence[tuple[Any, ...] | None],
    estimator_batch_size: int | None,
) -> list[slice]:
    # Consecutive estimators that can go through one `_forward` together,
    # split when shapes/dtypes or the target class set change, or the
    # batch is full.
    related = any(context.related_tables is not None for context in contexts)
    if queries is not None:
        related = related or any(
            query.related_tables is not None for query in queries
        )
    if related:
        return [slice(i, i + 1) for i in range(len(contexts))]

    batches: list[slice] = []
    start = 0
    key: Hashable = None
    # Stack when table shapes/dtypes match and the target class set matches.
    for i, context in enumerate(contexts):
        query = None if queries is None else queries[i]
        tables = (
            [context.x, context.y]
            if query is None
            else [context.x, context.y, query.x]
        )
        classes = class_values[i]
        member_key = (
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
            None if classes is None else frozenset(classes),
        )
        if i > start and (
            member_key != key
            or (
                estimator_batch_size is not None
                and i - start == estimator_batch_size
            )
        ):
            batches.append(slice(start, i))
            start = i
        key = member_key
    batches.append(slice(start, len(contexts)))
    return batches


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


def _class_values(
    contexts: Sequence[MemberContext],
    estimator_batch_size: int | None,
) -> list[tuple[Any, ...] | None]:
    # Class labels per estimator, used to group and relabel stacked batches.
    # Unused when each estimator already has its own ``_forward``.
    if (
        estimator_batch_size == 1
        or contexts[0].related_tables is not None
        or contexts[0].y.categorical.size(-1) == 0
    ):
        return [None] * len(contexts)
    return [
        tuple(context.y.categorical.categories[0].tolist())
        for context in contexts
    ]


def _unstack(
    out: TableTensor,
    class_values: Sequence[tuple[Any, ...] | None] | None,
    num_members: int,
) -> list[TableTensor]:
    outs = (
        [out]
        if num_members == 1
        else list(cast(tuple[TableTensor, ...], out.unbind(0)))
    )
    if class_values is None or len(class_values) == 1:
        return outs
    first = class_values[0]
    if first is None:
        return outs
    # The model labels columns in the first member's class order; member `e`'s
    # column `j` holds class `class_values[e][j]`.
    labels = out.columns[Stype.numerical]
    index = {value: i for i, value in enumerate(first)}
    stacked = cast(Sequence[tuple[Any, ...]], class_values)
    return [
        TableTensor(
            columns={Stype.numerical: [labels[index[v]] for v in classes]},
            numerical=member_out.numerical,
        )
        for member_out, classes in zip(outs, stacked, strict=True)
    ]
