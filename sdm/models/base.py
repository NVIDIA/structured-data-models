import abc
import copy
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    Recipe,
    RelatedTables,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm._inference import inference_mode
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.processing.execution import (
    ContextGroup,
    MemberContext,
    MemberQuery,
    QueryGroup,
    RecipeExecution,
)
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema

_CachedGroup = tuple[tuple[int, ...], Cache]


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

    #: Supported ensemble execution modes.
    supported_execution_modes: ClassVar[frozenset[str]] = frozenset(
        {"sequential"}
    )

    def __init__(self, *, estimator_execution: str = "sequential") -> None:
        super().__init__()
        self.estimator_execution = estimator_execution
        self._cache: Cache | None = None
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
        group_members = self._uses_grouped_execution()
        with torch.amp.autocast(x_query.device.type, enabled=False):
            context_groups = recipe_execution.fit_transform(
                x=x_context,
                y=y_context,
                related_tables=related_context_tables,
                num_members=num_estimators,
                group_members=group_members,
                generator=generator,
            )
            query_groups = recipe_execution.transform(
                x=x_query,
                related_tables=related_query_tables,
            )

        contexts: list[MemberContext | None] = [None] * num_estimators
        queries: list[MemberQuery | None] = [None] * num_estimators
        outs: list[TableTensor | None] = [None] * num_estimators
        for context_group, query_group in zip(
            context_groups, query_groups, strict=True
        ):
            if context_group.member_ids != query_group.member_ids:
                raise RuntimeError("Context and query groups do not match")
            prepared_queries = []
            for member_id, context, query in zip(
                context_group.member_ids,
                context_group.members,
                query_group.members,
                strict=True,
            ):
                self._validate_context(
                    x=context.x,
                    y=context.y,
                    related_tables=context.related_tables,
                )
                query_x = query.x
                query_related_tables = query.related_tables
                for callback in callbacks:
                    query_x, query_related_tables = (
                        callback.on_preprocessing_end(
                            self,
                            query_x,
                            query_related_tables,
                        )
                    )
                query = replace(
                    query,
                    x=query_x,
                    related_tables=query_related_tables,
                )
                self._validate_query(
                    x_context=context.x.schema,
                    x_query=query_x,
                    related_context_tables=context.related_tables.schema
                    if context.related_tables is not None
                    else None,
                    related_query_tables=query_related_tables,
                )
                contexts[member_id] = context
                queries[member_id] = query
                prepared_queries.append(query)

            query_group = replace(
                query_group,
                members=tuple(prepared_queries),
            )

            group_outs = self._forward_group(
                context_group=context_group,
                query_group=query_group,
                cache=None,
                generator=generator,
                **kwargs,
            )
            for member_id, out, query in zip(
                context_group.member_ids,
                group_outs,
                query_group.members,
                strict=True,
            ):
                outs[member_id] = cast(TableTensor, out.to(query.x.dtype))

        if any(item is None for item in (*contexts, *queries, *outs)):
            raise RuntimeError("Expected one result per estimator")
        contexts_out = tuple(cast(MemberContext, item) for item in contexts)
        outs_out = tuple(cast(TableTensor, item) for item in outs)

        # Regression: invert target before stacking estimator outputs.
        if contexts_out[0].y.numerical.size(-1) > 0:
            with torch.amp.autocast(x_query.device.type, enabled=False):
                outs_out = recipe_execution.inverse_transform_target(outs_out)

        with torch.amp.autocast(x_query.device.type, enabled=False):
            prediction = recipe_execution.transform_output(outs_out)

        for callback in callbacks:
            callback.on_forward_end(self, prediction)

        return prediction

    @inference_mode(False)
    @torch.no_grad()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
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

        self.clear()

        recipe_execution = RecipeExecution(
            self.default_recipe() if recipe is None else copy.deepcopy(recipe)
        )
        group_members = self._uses_grouped_execution()
        with torch.amp.autocast(x.device.type, enabled=False):
            context_groups = recipe_execution.fit_transform(
                x=x,
                y=y,
                related_tables=related_tables,
                num_members=num_estimators,
                group_members=group_members,
                generator=generator,
            )

        cache = Cache(
            num_estimators=num_estimators,
            recipe_execution=recipe_execution,
            kwargs=kwargs,
        )
        cached_groups: list[_CachedGroup] = []
        for context_group in context_groups:
            metadata = []
            for context in context_group.members:
                self._validate_context(
                    x=context.x,
                    y=context.y,
                    related_tables=context.related_tables,
                )
                metadata.append(self._member_metadata(context))

            model_cache = (
                metadata[0]
                if len(context_group.member_ids) == 1
                else Cache(classes=metadata[0]["classes"])
            )
            self._forward_group(
                context_group=context_group,
                query_group=None,
                cache=model_cache,
                generator=generator,
                **kwargs,
            )

            if x.is_cuda and num_estimators > 1:
                model_cache = model_cache.cpu().pin_memory()
                if len(context_group.member_ids) == 1:
                    metadata = [model_cache]
                else:
                    metadata = [item.cpu().pin_memory() for item in metadata]
            for member_id, item in zip(
                context_group.member_ids, metadata, strict=True
            ):
                cache[member_id] = item
            cached_groups.append((context_group.member_ids, model_cache))

        cache["member_groups"] = tuple(cached_groups)
        self._cache = cache.freeze()

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
            related_tables: Related context for query examples.
            callbacks: Callbacks applied in sequence to this model call.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
        callbacks = () if callbacks is None else callbacks
        requires_grad = any(callback.requires_grad for callback in callbacks)
        with inference_mode(not requires_grad):
            return self._predict_call(
                x=x,
                related_tables=related_tables,
                callbacks=callbacks,
            )

    def _predict_call(
        self,
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
        cached_groups = cast(
            tuple[_CachedGroup, ...], self._cache["member_groups"]
        )
        next_cache = cached_groups[0][1]

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
                query_groups = recipe_execution.transform(x, related_tables)

            queries: list[MemberQuery | None] = [None] * num_estimators
            prepared_query_groups = []
            for query_group in query_groups:
                prepared_queries = []
                for member_id, query in zip(
                    query_group.member_ids, query_group.members, strict=True
                ):
                    metadata = cast(Cache, self._cache[member_id])
                    query_x = query.x
                    query_related_tables = query.related_tables
                    for callback in callbacks:
                        query_x, query_related_tables = (
                            callback.on_preprocessing_end(
                                self,
                                query_x,
                                query_related_tables,
                            )
                        )
                    query = replace(
                        query,
                        x=query_x,
                        related_tables=query_related_tables,
                    )
                    self._validate_query(
                        x_context=cast(TableSchema, metadata["x_schema"]),
                        x_query=query_x,
                        related_context_tables=cast(
                            RelatedTablesSchema | None,
                            metadata["related_tables_schema"],
                        ),
                        related_query_tables=query_related_tables,
                    )
                    queries[member_id] = query
                    prepared_queries.append(query)
                prepared_query_groups.append(
                    replace(query_group, members=tuple(prepared_queries))
                )
            query_groups = tuple(prepared_query_groups)

            if x.is_cuda:
                assert compute_stream is not None
                assert transfer_stream is not None
                compute_stream.wait_stream(transfer_stream)

            outs: list[TableTensor | None] = [None] * num_estimators
            for group_id, (query_group, cached_group) in enumerate(
                zip(query_groups, cached_groups, strict=True)
            ):
                member_ids, _ = cached_group
                if query_group.member_ids != member_ids:
                    raise RuntimeError("Cached estimator groups do not match")
                cache, next_cache = next_cache, None
                assert cache is not None

                if group_id + 1 < len(cached_groups):
                    next_cache = cached_groups[group_id + 1][1]
                if x.is_cuda and next_cache is not None:
                    assert transfer_stream is not None
                    with torch.cuda.stream(transfer_stream):
                        next_cache = next_cache.to(x.device, non_blocking=True)

                metadata = tuple(
                    cast(Cache, self._cache[member_id])
                    for member_id in member_ids
                )
                group_outs = self._predict_group(
                    query_group=query_group,
                    metadata=metadata,
                    cache=cache,
                    **cast(dict[str, Any], self._cache["kwargs"]),
                )

                if x.is_cuda:
                    assert compute_stream is not None
                    for tensor in cache._tensors():
                        tensor.record_stream(compute_stream)

                for member_id, out, query in zip(
                    member_ids,
                    group_outs,
                    query_group.members,
                    strict=True,
                ):
                    outs[member_id] = cast(TableTensor, out.to(query.x.dtype))

                if x.is_cuda and next_cache is not None:
                    assert compute_stream is not None
                    assert transfer_stream is not None
                    compute_stream.wait_stream(transfer_stream)

        except BaseException:
            if transfer_stream is not None:
                transfer_stream.synchronize()
            raise

        if any(item is None for item in (*queries, *outs)):
            raise RuntimeError("Expected one result per estimator")
        outs_out = [cast(TableTensor, item) for item in outs]

        # Regression: invert target before stacking estimator outputs.
        if cast(Cache, self._cache[0])["classes"] is None:
            with torch.amp.autocast(x.device.type, enabled=False):
                outs_out = list(
                    recipe_execution.inverse_transform_target(outs_out)
                )

        with torch.amp.autocast(x.device.type, enabled=False):
            prediction = recipe_execution.transform_output(outs_out)

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
        pass

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    # Helpers #################################################################

    def _uses_grouped_execution(self) -> bool:
        if self.estimator_execution not in self.supported_execution_modes:
            supported = ", ".join(
                repr(mode) for mode in sorted(self.supported_execution_modes)
            )
            raise ValueError(
                f"Unsupported estimator execution "
                f"{self.estimator_execution!r}; expected one of {supported}"
            )
        return self.estimator_execution != "sequential"

    @staticmethod
    def _member_metadata(context: MemberContext) -> Cache:
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

    def _forward_group(
        self,
        context_group: ContextGroup,
        query_group: QueryGroup | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> tuple[TableTensor, ...]:
        contexts = context_group.members
        x_context = self._stack_tables(
            tuple(context.x for context in contexts)
        )
        y_context = self._stack_tables(
            tuple(context.y for context in contexts)
        )
        related_context = self._stack_related_tables(
            tuple(context.related_tables for context in contexts)
        )
        x_query = None
        related_query = None
        if query_group is not None:
            x_query = self._stack_tables(
                tuple(query.x for query in query_group.members)
            )
            related_query = self._stack_related_tables(
                tuple(query.related_tables for query in query_group.members)
            )

        out = self._forward(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context,
            related_query_tables=related_query,
            cache=cache,
            generator=generator,
            **kwargs,
        )
        if query_group is None:
            return ()
        metadata = tuple(
            self._member_metadata(context) for context in contexts
        )
        return self._split_group_output(out, metadata)

    def _predict_group(
        self,
        query_group: QueryGroup,
        metadata: Sequence[Cache],
        cache: Cache,
        **kwargs: Any,
    ) -> tuple[TableTensor, ...]:
        x_query = self._stack_tables(
            tuple(query.x for query in query_group.members)
        )
        out = self._forward(
            x_context=None,
            y_context=None,
            x_query=x_query,
            related_context_tables=None,
            related_query_tables=self._stack_related_tables(
                tuple(query.related_tables for query in query_group.members)
            ),
            cache=cache,
            generator=None,
            **kwargs,
        )
        return self._split_group_output(out, metadata)

    @staticmethod
    def _stack_tables(tables: Sequence[TableTensor]) -> TableTensor:
        if len(tables) == 1:
            return tables[0]

        first = tables[0]
        blocks: dict[Stype, Tensor] = {}
        for stype, first_block in first.items():
            member_blocks = tuple(table.blocks[stype] for table in tables)
            if isinstance(first_block, CategoricalTensor):
                categorical_blocks = tuple(
                    cast(CategoricalTensor, block) for block in member_blocks
                )
                blocks[stype] = CategoricalTensor(
                    code=torch.stack(
                        tuple(block.code for block in categorical_blocks),
                        dim=0,
                    ),
                    categories=first_block.categories,
                )
            else:
                blocks[stype] = torch.stack(member_blocks, dim=0)
        return first.replace_blocks(
            numerical=blocks[Stype.numerical],
            categorical=cast(CategoricalTensor, blocks[Stype.categorical]),
            datetime=blocks[Stype.datetime],
            text=cast(StringTensor, blocks[Stype.text]),
            id=cast(ColumnarTensor, blocks[Stype.id]),
        )

    @classmethod
    def _stack_related_tables(
        cls,
        related: Sequence[RelatedTables | None],
    ) -> RelatedTables | None:
        first = related[0]
        if first is None:
            return None
        if len(related) == 1:
            return first
        return replace(
            first,
            tables={
                name: cls._stack_tables(
                    tuple(
                        cast(RelatedTables, item).tables[name]
                        for item in related
                    )
                )
                for name in first.tables
            },
        )

    @staticmethod
    def _split_group_output(
        output: TableTensor,
        metadata: Sequence[Cache],
    ) -> tuple[TableTensor, ...]:
        if len(metadata) == 1:
            return (output,)
        outputs = tuple(
            cast(TableTensor, item) for item in output.unbind(dim=0)
        )
        restored = []
        for item, member_metadata in zip(outputs, metadata, strict=True):
            classes = cast(Tensor | None, member_metadata["classes"])
            if classes is not None:
                item = TableTensor(
                    columns={
                        Stype.numerical: tuple(
                            str(value) for value in classes.tolist()
                        )
                    },
                    numerical=item.numerical[..., : len(classes)],
                )
            restored.append(item)
        return tuple(restored)

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
