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
from sdm.models.context import CompiledContext
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
        self._cache: CompiledContext | None = None
        self._context_owner_token = object()
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}

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
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = any(callback.requires_grad for callback in callbacks)
        with inference_mode(not requires_grad):
            return self._forward_call(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                related_context_tables=related_context_tables,
                related_query_tables=related_query_tables,
                recipe=recipe,
                num_estimators=num_estimators,
                generator=generator,
                callbacks=callbacks,
                **kwargs,
            )

    def _forward_call(
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
        callbacks: Sequence[Callback],
        **kwargs: Any,
    ) -> TableTensor:
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

    @inference_mode(False)
    @torch.no_grad()
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
        r"""Fit context into the model's backward-compatible cache slot.

        Existing context state is replaced only after compilation succeeds.
        Use :meth:`compile_context` to retain multiple independent contexts.
        """
        replacement = self.compile_context(
            x=x,
            y=y,
            related_tables=related_tables,
            recipe=recipe,
            num_estimators=num_estimators,
            generator=generator,
            **kwargs,
        )
        previous, self._cache = self._cache, replacement
        if previous is not None:
            previous.close()

    @inference_mode(False)
    @torch.no_grad()
    def compile_context(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> CompiledContext:
        r"""Compile reusable context state without changing the model.

        Repeated calls to :meth:`predict_context` can reuse the same in-context
        examples while only providing new query examples.

        Args:
            x: The feature tensor of in-context examples with shape
                ``[..., R, D]`` with ``R`` rows and ``C`` columns.
            y: The targets of in-context examples with shape
                ``[..., R, 1]``.
            related_tables: Related context for in-context examples.
            recipe: The recipe for pre- and post-processing. If ``None``, the
                model's default recipe is applied.
            num_estimators: The number of estimators for ensembling.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.
        """
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive")
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)

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

        cache = cache.freeze()
        first_cache = cache[0]
        assert isinstance(first_cache, Cache)
        task = (
            "classification"
            if first_cache["classes"] is not None
            else "regression"
        )
        return CompiledContext(
            recipe_execution=recipe_execution,
            cache=cache,
            owner_token=self._context_owner_token,
            weight_versions=self._weight_versions(),
            submodel_token=self._submodel_token(task),
            task=task,
        )

    def predict(
        self,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        r"""Predict with context stored by :meth:`fit`."""
        callbacks = () if callbacks is None else callbacks
        requires_grad = any(callback.requires_grad for callback in callbacks)
        with inference_mode(not requires_grad):
            return self._predict_context_call(
                context=self._cache,
                x=x,
                related_tables=related_tables,
                callbacks=callbacks,
            )

    def predict_context(
        self,
        context: CompiledContext,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        r"""Predict unseen query examples with a compiled context.

        Args:
            context: Compiled context returned by :meth:`compile_context`.
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.
            callbacks: Callbacks applied in sequence to this model call.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = any(callback.requires_grad for callback in callbacks)
        with inference_mode(not requires_grad):
            return self._predict_context_call(
                context=context,
                x=x,
                related_tables=related_tables,
                callbacks=callbacks,
            )

    def _predict_context_call(
        self,
        context: CompiledContext | None,
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback],
    ) -> TableTensor:
        for callback in callbacks:
            callback.on_forward_start(
                self,
                x,
                related_tables,
                callbacks=callbacks,
            )
        if context is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call {self.__class__.__name__}.fit() before."
            )
        self._validate_compiled_context(context)
        recipe_execution, compiled_cache = context._state()

        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

        if related_tables is not None:
            first_cache = compiled_cache[0]
            assert isinstance(first_cache, Cache)
            if first_cache["related_tables_schema"] is None:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(
                tables=cast(
                    RelatedTablesSchema,
                    first_cache["related_tables_schema"],
                ).tables,
            )

        num_estimators = cast(int, compiled_cache["num_estimators"])
        caches = [
            cast(Cache, compiled_cache[i]) for i in range(num_estimators)
        ]
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
                    **cast(dict[str, Any], compiled_cache["kwargs"]),
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
        first_cache = compiled_cache[0]
        assert isinstance(first_cache, Cache)
        if first_cache["classes"] is None:
            with torch.amp.autocast(x.device.type, enabled=False):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with torch.amp.autocast(x.device.type, enabled=False):
            prediction = recipe_execution.transform_output(outs)

        for callback in callbacks:
            callback.on_forward_end(self, prediction)

        return prediction

    def clear(self) -> None:
        r"""Close and clear context state created by :meth:`fit`."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def _weight_versions(self) -> tuple[tuple[object, ...], ...]:
        tensors = (
            (f"parameter:{name}", tensor)
            for name, tensor in self.named_parameters()
        )
        buffers = (
            (f"buffer:{name}", tensor) for name, tensor in self.named_buffers()
        )
        return tuple(
            (
                name,
                id(tensor),
                tensor._version,
                tensor.device,
                tensor.dtype,
            )
            for name, tensor in (*tensors, *buffers)
        )

    def _submodel_token(self, task: str) -> int:
        if task == "classification":
            return id(getattr(self, "cls_model", self))
        assert task == "regression"
        return id(getattr(self, "reg_model", self))

    def _validate_compiled_context(self, context: CompiledContext) -> None:
        context._state()
        if context._owner_token is not self._context_owner_token:
            raise ValueError(
                "Compiled context belongs to a different model instance"
            )
        if context._weight_versions != self._weight_versions():
            raise RuntimeError(
                "Compiled context is stale because model weights or placement "
                "changed"
            )
        if context._submodel_token != self._submodel_token(context._task):
            raise RuntimeError(
                "Compiled context targets a different model submodel"
            )

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
