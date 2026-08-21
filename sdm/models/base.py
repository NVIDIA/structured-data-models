import abc
import copy
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm._inference import inference_mode
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.processing.execution import RecipeExecution
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema


class ICLModel(torch.nn.Module, abc.ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`ICLModel` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    #: Semantic types supported for input columns in this model.
    supported_feature_stypes: ClassVar[frozenset[Stype]]

    #: Semantic types supported for target columns in this model.
    supported_target_stypes: ClassVar[frozenset[Stype]]

    #: Whether this model supports additional related context.
    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()
        self._cache: Cache | None = None
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}

    @inference_mode()
    def forward(
        self,
        x_context: Tensor | TableTensor,  # [..., R_context, D]
        y_context: Tensor | TableTensor,  # [..., R_context, 1]
        x_query: Tensor | TableTensor,  # [..., R_query, D]
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        callbacks: Sequence[Callback] | None = None,
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
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            callbacks: Callbacks applied in sequence to this model call.
            seqused_train: Valid in-context example counts with shape
                ``[...]`` and :external+torch:ref:`torch.int32 <dtype-doc>`
                dtype.
                Out-of-range counts are clamped and produce degenerate
                predictions rather than errors. When set, only the first
                ``seqused_train`` of the ``R_context`` in-context rows act as
                context, and the remaining rows are treated as padding: they
                are masked from every attention key/value stream and cannot
                influence any prediction, provided the padded feature
                entries are finite and of moderate magnitude (``0`` is
                recommended - masking adds ``-inf`` to attention logits
                after the query/key product, so non-finite or overflowing
                padded values poison the softmax with ``NaN``).
                Padded ``y_context`` entries must still be valid targets (for
                example ``0``). Together with padded query rows (whose extra
                outputs callers simply discard), this lets streams of varying
                table sizes be padded to a small set of bucketed shapes so
                compiled graphs and per-shape kernel selection are reused
                across tables. Pass a tensor rather than a Python integer so
                compiled graphs treat the count as data instead of a
                constant to specialize on.
                Requires a pass-through ``recipe`` since fitted pre-processing
                would derive its state from the padded rows.
            seqused_cols: Valid column count as a positive scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, only the first ``seqused_cols`` columns act as
                features; the remaining columns are padding, excluded from
                feature grouping and masked from row-wise attention. The
                count refers to the numerical block handed to the model
                (``x.numerical.size(-1)``), not the table width -
                non-numerical columns are removed before masking applies.
                The same finite-values contract as ``seqused_train``
                applies; out-of-range counts are clamped and produce
                degenerate predictions rather than errors.
                Unlike ``seqused_train``, the count is shared across batch
                elements.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        callbacks = () if callbacks is None else callbacks
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

        self._validate_seqused(seqused_train, seqused_cols)
        self._warn_seqused_cols_table(x_context, seqused_cols)
        self._validate_seqused_recipe(recipe, seqused_train, seqused_cols)

        # Only forward the padding keywords when set so that subclasses
        # implementing a narrower private hook keep working.
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols

        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(y_context, TableTensor):
            y_context = TableTensor.from_tensor(y_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)

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
        with torch.amp.autocast(x_query.device.type, enabled=False):
            contexts = recipe_execution.fit_transform(
                x=x_context,
                y=y_context,
                related_tables=related_context_tables,
                num_members=num_estimators,
                generator=generator,
            )
            queries = recipe_execution.transform(
                x=x_query,
                related_tables=related_query_tables,
            )

        outs: list[TableTensor] = []
        for context, query in zip(contexts, queries):
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )
            x_query = query.x
            related_query_tables = query.related_tables
            for callback in callbacks:
                x_query, related_query_tables = callback.on_preprocessing_end(
                    self,
                    x_query,
                    related_query_tables,
                )
            self._validate_query(
                x_context=context.x.schema,
                x_query=x_query,
                related_context_tables=context.related_tables.schema
                if context.related_tables is not None
                else None,
                related_query_tables=related_query_tables,
            )
            out = self._forward(
                x_context=context.x,
                y_context=context.y,
                x_query=x_query,
                related_context_tables=context.related_tables,
                related_query_tables=related_query_tables,
                cache=None,
                generator=generator,
                **kwargs,
            )
            out = cast(TableTensor, out.to(x_query.dtype))
            outs.append(out)

        # Regression: invert target before stacking estimator outputs.
        if contexts[0].y.numerical.size(-1) > 0:
            with torch.amp.autocast(x_query.device.type, enabled=False):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with torch.amp.autocast(x_query.device.type, enabled=False):
            prediction = recipe_execution.transform_output(outs)

        for callback in callbacks:
            callback.on_forward_end(self, prediction)

        return prediction

    @inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
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
            num_estimators: The number of estimators for ensembling.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            seqused_train: Valid in-context example counts with shape
                ``[...]`` and :external+torch:ref:`torch.int32 <dtype-doc>`
                dtype.
                When set, rows beyond the per-element count are padding and
                are masked from the cached key/value projections; subsequent
                :meth:`predict` calls reuse the count automatically. See
                :meth:`forward` for the padding contract.
            seqused_cols: Valid column count as a scalar tensor with
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
                When set, columns beyond the count are padding; subsequent
                :meth:`predict` calls reuse the count and must pass ``x``
                padded to the same number of columns. See :meth:`forward`
                for the padding contract.
            kwargs: Additional keyword arguments passed to the model. They
                are cached and re-applied by :meth:`predict`.
        """
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")

        self._validate_seqused(seqused_train, seqused_cols)
        self._warn_seqused_cols_table(x, seqused_cols)
        self._validate_seqused_recipe(recipe, seqused_train, seqused_cols)

        # Only forward the padding keywords when set so that subclasses
        # implementing a narrower private hook keep working. The counts
        # become part of the cached keyword arguments, so :meth:`predict`
        # replays them alongside the cached key/value projections.
        if seqused_train is not None:
            kwargs["seqused_train"] = seqused_train
        if seqused_cols is not None:
            kwargs["seqused_cols"] = seqused_cols

        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)

        self.clear()

        recipe_execution = RecipeExecution(
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        with torch.amp.autocast(x.device.type, enabled=False):
            contexts = recipe_execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                generator=generator,
            )

        cache = Cache(
            num_estimators=num_estimators,
            recipe_execution=recipe_execution,
            kwargs=kwargs,
        )
        for i, context in enumerate(contexts):
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )
            estimator_cache = Cache(
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
            if x.is_cuda and num_estimators > 1:
                estimator_cache = estimator_cache.cpu().pin_memory()
            cache[i] = estimator_cache

        self._cache = cache.freeze()

    @inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
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

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        for callback in callbacks:
            callback.on_forward_start(
                self,
                x,
                related_tables,
                callbacks=callbacks,
            )

        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

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

        num_estimators = cast(int, self._cache["num_estimators"])
        caches = [cast(Cache, self._cache[i]) for i in range(num_estimators)]
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

            recipe_execution = cast(
                RecipeExecution,
                self._cache["recipe_execution"],
            )
            with torch.amp.autocast(x.device.type, enabled=False):
                queries = recipe_execution.transform(x, related_tables)

            if x.is_cuda:
                assert compute_stream is not None
                assert transfer_stream is not None
                compute_stream.wait_stream(transfer_stream)

            outs: list[TableTensor] = []
            for i, query in enumerate(queries):
                cache, next_cache = next_cache, None
                assert cache is not None

                x_query = query.x
                related_query_tables = query.related_tables
                for callback in callbacks:
                    x_query, related_query_tables = (
                        callback.on_preprocessing_end(
                            self,
                            x_query,
                            related_query_tables,
                        )
                    )
                self._validate_query(
                    x_context=cast(TableSchema, cache["x_schema"]),
                    x_query=x_query,
                    related_context_tables=cast(
                        RelatedTablesSchema,
                        cache["related_tables_schema"],
                    ),
                    related_query_tables=related_query_tables,
                )

                if i + 1 < num_estimators:
                    next_cache = caches[i + 1]
                if x.is_cuda and next_cache is not None:
                    assert transfer_stream is not None
                    with torch.cuda.stream(transfer_stream):
                        next_cache = next_cache.to(x.device, non_blocking=True)

                out = self._forward(
                    x_context=None,
                    y_context=None,
                    x_query=x_query,
                    related_context_tables=None,
                    related_query_tables=related_query_tables,
                    cache=cache,
                    generator=None,
                    **cast(dict[str, Any], self._cache["kwargs"]),
                )

                if x.is_cuda:
                    assert compute_stream is not None
                    for tensor in cache._tensors():
                        tensor.record_stream(compute_stream)

                out = cast(TableTensor, out.to(x_query.dtype))
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
            with torch.amp.autocast(x.device.type, enabled=False):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with torch.amp.autocast(x.device.type, enabled=False):
            prediction = recipe_execution.transform_output(outs)

        for callback in callbacks:
            callback.on_forward_end(self, prediction)

        return prediction

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
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        # Padding keywords such as `seqused_train`/`seqused_cols` (and model
        # specific ones such as `batch_size_limit`) only reach `kwargs` when
        # the caller sets them, so subclasses that do not consume them keep
        # working.
        pass

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _validate_context(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables | None,
    ) -> None:

        if y.size(-1) != 1:
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
        related_query_tables: RelatedTables | None,
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
        seqused_cols: Tensor | None = None,
    ) -> None:
        # Only dtypes are checked. Validating the counts themselves would
        # read them off the device on every call, and that synchronization
        # would land directly in the serving latencies this padding exists
        # to improve. Out-of-range counts are clamped where they are
        # consumed instead, matching `seqused_cols`.
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

    def _validate_seqused_recipe(
        self,
        recipe: Recipe | None,
        seqused_train: Tensor | None,
        seqused_cols: Tensor | None = None,
    ) -> None:
        if seqused_train is None and seqused_cols is None:
            return

        # A fitted recipe derives its state from every context row and
        # column, including the padded ones, which lets padding influence
        # predictions in violation of the seqused contract. Only
        # pass-through pre-processing may be combined with padded inputs.
        effective = self.default_recipe() if recipe is None else recipe
        if effective.features.requires_fit or effective.target.requires_fit:
            raise ValueError(
                "Recipe pre-processing fits its state on the padded rows "
                "and targets, letting padding influence predictions in "
                "violation of the seqused contract; pass a pass-through "
                "recipe such as 'sdm.processing.Recipe()' together with "
                "'seqused_train'/'seqused_cols'"
            )

    def _warn_seqused_cols_table(
        self,
        x: Tensor | TableTensor,
        seqused_cols: Tensor | None,
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
                    f"numerical block ({x.numerical.size(-1)} columns), but "
                    f"'x' has {x.size(-1)} table columns; non-numerical "
                    f"columns (including id data) are removed before "
                    f"masking applies."
                ),
                stacklevel=3,
            )
