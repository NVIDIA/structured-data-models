import abc
import copy
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import (
    EnsembleTable,
    Recipe,
    RelatedTables,
    Stype,
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

    #: Whether :meth:`forward`/:meth:`fit` accept padded inputs via
    #: ``seqused_train``/``seqused_cols``; models without support reject
    #: the keywords with a ``ValueError``.
    supports_seqused: ClassVar[bool] = False

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
        requires_grad = any(callback.requires_grad for callback in callbacks)

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

        seqused = seqused_train is not None or seqused_cols is not None

        outs: list[TableTensor] = []
        for context, query in zip(contexts, queries, strict=True):
            for callback in callbacks:
                context = self._apply_context_callback(
                    callback=callback,
                    context=context,
                    seqused=seqused,
                )
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )

            for callback in callbacks:
                query = self._apply_query_callback(
                    callback=callback,
                    query=query,
                    seqused_cols=seqused_cols is not None,
                )
            self._validate_query(
                x_context=context.x.schema,
                x_query=query.x,
                related_context_tables=context.related_tables.schema
                if context.related_tables is not None
                else None,
                related_query_tables=query.related_tables,
            )

            with inference_mode("grad" if requires_grad else "inference"):
                out = self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=query.x,
                    related_context_tables=context.related_tables,
                    related_query_tables=query.related_tables,
                    cache=None,
                    generator=generator,
                    **kwargs,
                )

                for callback in callbacks:
                    out = callback.on_model_forward_end(self, out)

            out = cast(TableTensor, out.to(query.x.dtype))
            outs.append(out)

        # Regression: invert target before stacking estimator outputs.
        if contexts[0].y.numerical.size(-1) > 0:
            with (
                torch.amp.autocast(x_query.device.type, enabled=False),
                inference_mode(),
            ):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with (
            torch.amp.autocast(x_query.device.type, enabled=False),
            inference_mode(),
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

        cache = Cache(
            recipe_execution=recipe_execution,
            kwargs=kwargs,
        )
        for i, context in enumerate(contexts):
            for callback in callbacks:
                context = self._apply_context_callback(
                    callback=callback,
                    context=context,
                    seqused=seqused_train is not None
                    or seqused_cols is not None,
                )
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )
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
            )

            with inference_mode("no_grad"):
                self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=None,
                    related_context_tables=context.related_tables,
                    related_query_tables=None,
                    cache=estimator_cache,
                    generator=generator,
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
        caches = [
            cast(Cache, self._cache[i])
            for i in range(recipe_execution.num_members)
        ]
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
            for i, query in enumerate(queries):
                cache, next_cache = next_cache, None
                assert cache is not None

                for callback in callbacks:
                    query = self._apply_query_callback(
                        callback=callback,
                        query=query,
                        seqused_cols=seqused_cols,
                    )
                self._validate_query(
                    x_context=cast(TableSchema, cache["x_schema"]),
                    x_query=query.x,
                    related_context_tables=cast(
                        RelatedTablesSchema,
                        cache["related_tables_schema"],
                    ),
                    related_query_tables=query.related_tables,
                )

                if i + 1 < len(caches):
                    next_cache = caches[i + 1]
                if x.is_cuda and next_cache is not None:
                    assert transfer_stream is not None
                    with torch.cuda.stream(transfer_stream):
                        next_cache = next_cache.to(x.device, non_blocking=True)

                with inference_mode("grad" if requires_grad else "inference"):
                    out = self._forward(
                        x_context=None,
                        y_context=None,
                        x_query=query.x,
                        related_context_tables=None,
                        related_query_tables=query.related_tables,
                        cache=cache,
                        generator=None,
                        **kwargs,
                    )

                    for callback in callbacks:
                        out = callback.on_model_forward_end(self, out)

                if x.is_cuda:
                    assert compute_stream is not None
                    for tensor in cache._tensors():
                        tensor.record_stream(compute_stream)

                out = cast(TableTensor, out.to(query.x.dtype))
                outs.append(out)

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
                inference_mode(),
            ):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with (
            torch.amp.autocast(x.device.type, enabled=False),
            inference_mode(),
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
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        # `seqused_train`/`seqused_cols` only reach `kwargs` when set, so
        # subclasses that do not consume them keep working.
        pass

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _apply_context_callback(
        self,
        callback: Callback,
        context: MemberContext,
        *,
        seqused: bool,
    ) -> MemberContext:
        x, y, related_tables = callback.on_context_preprocessing_end(
            self,
            *context,
        )
        # The counts identify padding by position, and a replacement context
        # may reorder rows or columns while keeping shape and schema intact,
        # so replacements are rejected by identity when counts are set. No
        # tensor data is read, so passive observers keep working without a
        # device sync; in-place mutation inside a hook is not detectable.
        if seqused and (
            x is not context.x
            or y is not context.y
            or related_tables is not context.related_tables
        ):
            raise ValueError(
                f"Callback {callback.__class__.__name__!r} returned a "
                f"replacement context, which cannot be verified to keep the "
                f"padded trailing rows and columns counted by "
                f"'seqused_train'/'seqused_cols' in place; use callbacks "
                f"that return the context unchanged together with "
                f"'seqused_train'/'seqused_cols'"
            )
        return MemberContext(x=x, y=y, related_tables=related_tables)

    def _apply_query_callback(
        self,
        callback: Callback,
        query: MemberQuery,
        *,
        seqused_cols: bool,
    ) -> MemberQuery:
        x, related_tables = callback.on_query_preprocessing_end(
            self,
            *query,
        )
        # Same identity check as `_apply_context_callback`, for `seqused_cols`
        # only: `seqused_train` counts context rows, which a replaced query
        # cannot invalidate (see `_GradientCallback`).
        if seqused_cols and (
            x is not query.x or related_tables is not query.related_tables
        ):
            raise ValueError(
                f"Callback {callback.__class__.__name__!r} returned a "
                f"replacement query, which cannot be verified to keep the "
                f"padded trailing columns counted by 'seqused_cols' in "
                f"place; use callbacks that return the query unchanged "
                f"together with 'seqused_cols'"
            )
        return MemberQuery(x=x, related_tables=related_tables)

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
        # Runs ahead of `clear()` in `fit` and ahead of pre-processing in
        # `forward`, mirroring the `supports_related_tables` check.
        if (
            seqused_train is not None or seqused_cols is not None
        ) and not self.supports_seqused:
            raise ValueError(
                f"{self.__class__.__name__!r} does not support the "
                f"'seqused_train'/'seqused_cols' padding keywords"
            )
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
