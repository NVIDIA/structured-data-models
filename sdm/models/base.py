# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import copy
from collections.abc import Iterable, Sequence
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

_CACHE_METADATA_KEYS = {
    "x_schema",
    "x_schemas",
    "y_schema",
    "related_tables_schema",
    "classes",
    "output_columns",
}


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
            callbacks: Callbacks applied in sequence to this model call.
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

        outs: list[TableTensor] = []
        for context, query in zip(contexts, queries):
            for callback in callbacks:
                context = MemberContext(
                    *callback.on_context_preprocessing_end(self, *context)
                )
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )

            for callback in callbacks:
                query = MemberQuery(
                    *callback.on_query_preprocessing_end(self, *query)
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
            estimator_batch_size: Maximum number of estimators per model
                execution. ``None`` runs all estimators together; ``1`` runs
                them sequentially. Batched estimators must have compatible
                shapes, target columns, and feature categories. Larger batches
                use more device memory.
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

        if estimator_batch_size is None:
            estimator_batch_size = len(contexts)
        starts = range(0, len(contexts), estimator_batch_size)
        cache = Cache(
            recipe_execution=recipe_execution,
            kwargs=kwargs,
            estimator_batch_size=estimator_batch_size,
        )
        for i, start in enumerate(starts):
            members = contexts[start : start + estimator_batch_size]
            with inference_mode("no_grad"):
                context = _stack_contexts(members)
            for callback in callbacks:
                context = MemberContext(
                    *callback.on_context_preprocessing_end(self, *context)
                )
            self._validate_context(
                x=context.x,
                y=context.y,
                related_tables=context.related_tables,
            )
            estimator_cache = Cache(
                x_schema=context.x.schema,
                x_schemas=_batch_schemas(members, context.x.schema),
                output_columns=_output_columns(members, context.y),
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

            if x.is_cuda and len(starts) > 1:
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
            estimator_batch_size: Maximum number of estimators per model
                execution. ``None`` runs all estimators together; ``1`` runs
                them sequentially. This can differ from the batch size used
                for :meth:`fit` when the model uses compatible tensor caches.
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
        if estimator_batch_size is None:
            estimator_batch_size = recipe_execution.num_members
        starts = range(0, recipe_execution.num_members, estimator_batch_size)
        with inference_mode("no_grad" if requires_grad else "inference"):
            caches = _prediction_caches(self._cache, estimator_batch_size)
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
            for i, start in enumerate(starts):
                cache, next_cache = next_cache, None
                assert cache is not None
                members = queries[start : start + estimator_batch_size]
                with inference_mode(
                    "no_grad" if requires_grad else "inference"
                ):
                    query = _stack_queries(members)

                for callback in callbacks:
                    query = MemberQuery(
                        *callback.on_query_preprocessing_end(self, *query)
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
                if len(members) > 1 and cache["x_schemas"] != _batch_schemas(
                    members, query.x.schema
                ):
                    raise ValueError(
                        "Expected context and query features to share "
                        "the same schema"
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
                        **cast(dict[str, Any], self._cache["kwargs"]),
                    )

                    output_columns = cast(
                        tuple[tuple[str, ...], ...] | None,
                        cache["output_columns"],
                    )
                    if callbacks and output_columns is not None:
                        outputs = _unstack_output(
                            out=out,
                            num_members=len(members),
                            columns=output_columns,
                        )
                        out = outputs[0]
                        if len(outputs) > 1:
                            names = out.columns[Stype.numerical]
                            # Align raw tensors to preserve gradients.
                            values: list[Tensor] = []
                            for output in outputs:
                                columns = output.columns[Stype.numerical]
                                indices = [
                                    columns.index(name) for name in names
                                ]
                                values.append(output.numerical[..., indices])
                            out = TableTensor(
                                columns={Stype.numerical: names},
                                numerical=torch.stack(values),
                            )
                        output_columns = None
                    for callback in callbacks:
                        out = callback.on_model_forward_end(self, out)

                if x.is_cuda:
                    assert compute_stream is not None
                    for tensor in cache._tensors():
                        tensor.record_stream(compute_stream)

                out = cast(TableTensor, out.to(query.x.dtype))
                with inference_mode("grad" if requires_grad else "inference"):
                    outs.extend(
                        _unstack_output(
                            out=out,
                            num_members=len(members),
                            columns=output_columns,
                        )
                    )

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
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]
        # Regroupable cache tensors retain all leading input batch dimensions.
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


def _stack_tables(
    tables: Sequence[TableTensor],
    *,
    target: bool = False,
) -> TableTensor:
    if len(tables) == 1:
        return tables[0]

    ref = tables[0]
    for table in tables[1:]:
        # Numerical feature positions can differ after column shuffling.
        # Other metadata must agree because the batch shares one schema.
        if table.active_stypes != ref.active_stypes or any(
            columns != ref.columns[stype]
            for stype, columns in table.columns.items()
            if target or stype != Stype.numerical
        ):
            raise ValueError(
                "Estimator batches require compatible column layouts; "
                "use 'estimator_batch_size=1' for incompatible estimators"
            )
        for categories, ref_categories in zip(
            table.categorical.categories,
            ref.categorical.categories,
            strict=True,
        ):
            compatible = (
                len(categories) == len(ref_categories)
                if target
                else categories is ref_categories
                or categories.equal(ref_categories)
            )
            if not compatible:
                raise ValueError(
                    "Estimator batches require matching class counts and "
                    "compatible feature categories; use "
                    "'estimator_batch_size=1' for incompatible estimators"
                )

    # TableTensor.stack aligns names, which would undo feature permutations.
    # Stack blocks by position; target class labels are restored on output.
    return TableTensor(
        columns=cast(dict[StypeLike, tuple[str, ...]], ref.columns),
        size=(len(tables), *ref.size()[:-1]),
        device=ref.device,
        **{
            stype: torch.stack(
                [table.blocks[stype] for table in tables], dim=0
            )
            for stype, _ in ref.items()
        },
    )


def _stack_related(
    tables: Sequence[RelatedTables[TableTensor] | None],
) -> RelatedTables[TableTensor] | None:
    ref = tables[0]
    if len(tables) == 1 or ref is None:
        return ref
    if any(table is None or table.schema != ref.schema for table in tables):
        raise ValueError(
            "Estimator batches require compatible related table schemas; "
            "use 'estimator_batch_size=1' for incompatible estimators"
        )
    tables = cast(Sequence[RelatedTables[TableTensor]], tables)
    return ref.replace_tables(
        {
            name: _stack_tables([table.tables[name] for table in tables])
            for name in ref.tables
        }
    )


def _stack_contexts(contexts: Sequence[MemberContext]) -> MemberContext:
    return MemberContext(
        x=_stack_tables([context.x for context in contexts]),
        y=_stack_tables([context.y for context in contexts], target=True),
        related_tables=_stack_related(
            [context.related_tables for context in contexts]
        ),
    )


def _stack_queries(queries: Sequence[MemberQuery]) -> MemberQuery:
    return MemberQuery(
        x=_stack_tables([query.x for query in queries]),
        related_tables=_stack_related(
            [query.related_tables for query in queries]
        ),
    )


def _remap_columns(
    columns: Sequence[tuple[str, ...]],
    before: tuple[str, ...],
    after: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if before == after:
        return tuple(columns)
    positions = {column: i for i, column in enumerate(before)}
    return tuple(
        tuple(
            names[positions[column]] if column in positions else column
            for column in after
        )
        for names in columns
    )


def _batch_schemas(
    members: Sequence[MemberContext] | Sequence[MemberQuery],
    schema: TableSchema,
) -> tuple[TableSchema, ...]:
    schemas = tuple(member.x.schema for member in members)
    if schemas[0] == schema:
        return schemas
    # Apply callback column selections/reordering to each estimator's layout.
    columns = _remap_columns(
        columns=[s.columns[Stype.numerical] for s in schemas],
        before=schemas[0].columns[Stype.numerical],
        after=schema.columns[Stype.numerical],
    )
    return tuple(
        TableSchema(columns={**schema.columns, Stype.numerical: names})
        for names in columns
    )


def _output_columns(
    contexts: Sequence[MemberContext],
    y: TableTensor,
) -> tuple[tuple[str, ...], ...] | None:
    if y.categorical.size(-1) == 0:
        return None
    names = tuple(str(value) for value in y.categorical.categories[0].tolist())
    if contexts[0].y.categorical.size(-1) == 0:
        return (names,) * len(contexts)
    columns = tuple(
        tuple(
            str(value)
            for value in context.y.categorical.categories[0].tolist()
        )
        for context in contexts
    )
    if len(names) == len(columns[0]) and set(names) != set(columns[0]):
        renamed = dict(zip(columns[0], names, strict=True))
        return tuple(
            tuple(renamed[name] for name in group) for group in columns
        )
    return _remap_columns(columns=columns, before=columns[0], after=names)


def _can_batch_cache(cache: Cache) -> bool:
    return all(
        isinstance(value, Tensor | KVCacheEntry)
        for estimator_cache in cache.values()
        if isinstance(estimator_cache, Cache)
        for key, value in estimator_cache.items()
        if key not in _CACHE_METADATA_KEYS
    )


def _prediction_caches(cache: Cache, estimator_batch_size: int) -> list[Cache]:
    num_members = cast(RecipeExecution, cache["recipe_execution"]).num_members
    fitted_estimator_batch_size = cast(int, cache["estimator_batch_size"])
    caches = [
        cast(Cache, cache[i])
        for i in range(len(range(0, num_members, fitted_estimator_batch_size)))
    ]
    if min(estimator_batch_size, num_members) == min(
        fitted_estimator_batch_size, num_members
    ):
        return caches

    members: list[Cache] = []
    for fitted in caches:
        schemas = cast(tuple[TableSchema, ...], fitted["x_schemas"])
        columns = cast(
            tuple[tuple[str, ...], ...] | None, fitted["output_columns"]
        )
        for index, schema in enumerate(schemas):
            member = Cache(fitted)
            member["x_schema"] = schema
            member["x_schemas"] = (schema,)
            member["output_columns"] = (
                None if columns is None else (columns[index],)
            )
            if len(schemas) > 1:
                for key, value in fitted.items():
                    if key not in _CACHE_METADATA_KEYS:
                        if isinstance(value, Tensor):
                            member[key] = value[index]
                        elif isinstance(value, KVCacheEntry):
                            member[key] = KVCacheEntry(
                                key=value.key[index], value=value.value[index]
                            )
                        else:
                            raise ValueError(
                                "Changing estimator batch size requires "
                                "tensor caches"
                            )
            members.append(member)

    batches: list[Cache] = []
    for start in range(0, num_members, estimator_batch_size):
        group = members[start : start + estimator_batch_size]
        first = group[0]
        if len(group) == 1:
            batches.append(first.freeze())
            continue
        classes = cast(Tensor | None, first["classes"])
        for member in group[1:]:
            other_classes = cast(Tensor | None, member["classes"])
            if (
                member.keys() != first.keys()
                or member["y_schema"] != first["y_schema"]
                or member["related_tables_schema"]
                != first["related_tables_schema"]
                or (classes is None) != (other_classes is None)
                or (
                    classes is not None
                    and len(classes) != len(cast(Tensor, other_classes))
                )
            ):
                raise ValueError(
                    "Estimator batches require compatible caches; "
                    "use 'estimator_batch_size=1'"
                )
        batched = Cache(first)
        batched["x_schemas"] = tuple(member["x_schema"] for member in group)
        batched["output_columns"] = (
            None
            if classes is None
            else tuple(
                cast(tuple[tuple[str, ...], ...], member["output_columns"])[0]
                for member in group
            )
        )
        for key, value in first.items():
            if key not in _CACHE_METADATA_KEYS:
                values = [member[key] for member in group]
                if isinstance(value, Tensor):
                    batched[key] = torch.stack(cast(list[Tensor], values))
                elif isinstance(value, KVCacheEntry):
                    entries = cast(list[KVCacheEntry], values)
                    batched[key] = KVCacheEntry(
                        key=torch.stack([entry.key for entry in entries]),
                        value=torch.stack([entry.value for entry in entries]),
                    )
                else:
                    raise ValueError(
                        "Changing estimator batch size requires tensor caches"
                    )
        batches.append(batched.freeze())
    return batches


def _unstack_output(
    out: TableTensor,
    num_members: int,
    columns: Sequence[tuple[str, ...]] | None,
) -> list[TableTensor]:
    outputs = (
        [out]
        if num_members == 1
        else list(cast(tuple[TableTensor, ...], out.unbind(0)))
    )
    if columns is None:
        return outputs
    return [
        TableTensor(
            columns={Stype.numerical: names},
            numerical=output.numerical,
        )
        for output, names in zip(outputs, columns, strict=True)
    ]


def _categorical_mask(
    x: TableTensor,
    schema: TableSchema,
    schemas: Sequence[TableSchema],
) -> Tensor:
    categorical_columns = set(schema.columns[Stype.categorical])
    mask = torch.tensor(
        [
            [
                column in categorical_columns
                for column in s.columns[Stype.numerical]
            ]
            for s in schemas
        ],
        device=x.device,
        dtype=torch.bool,
    )
    if len(schemas) == 1:
        return mask[0]
    # [E, C] -> [E, 1, ..., C], preserving existing input batch dimensions.
    return mask.view(len(schemas), *((1,) * (x.dim() - 3)), -1)
