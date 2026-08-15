import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.processing.execution import RecipeExecution
from sdm.relational.task import RelatedTablesSchema
from sdm.tensor.table import TableSchema


@contextlib.contextmanager
def _maybe_inference_mode() -> Iterator[None]:
    # `torch.inference_mode` is not supported inside a compiled region, so do
    # not enter it when this function is already being compiled.
    # https://github.com/pytorch/pytorch/issues/180823
    # FIXME: Come up with a solution to use torch.compile under
    # torch.inference_mode and remove this workaround.
    if torch.compiler.is_compiling():
        context_fn = contextlib.nullcontext
    else:
        context_fn = torch.inference_mode

    with context_fn():
        yield


class ICLModel(torch.nn.Module, ABC):
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
        self._cache_is_pinned = False
        self._cache_transfer_stream: torch.cuda.Stream | None = None

    @_maybe_inference_mode()
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
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R_query, *]``.
        """
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
            self._validate_query(
                x_context=context.x.schema,
                x_query=query.x,
                related_context_tables=context.related_tables.schema
                if context.related_tables is not None
                else None,
                related_query_tables=query.related_tables,
            )
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
            out = cast(TableTensor, out.to(query.x.dtype))
            outs.append(out)

        # Regression: invert target before stacking estimator outputs.
        if contexts[0].y.numerical.size(-1) > 0:
            with torch.amp.autocast(x_query.device.type, enabled=False):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with torch.amp.autocast(x_query.device.type, enabled=False):
            return recipe_execution.transform_output(outs)

    @_maybe_inference_mode()
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
        self._cache_is_pinned = x.is_cuda and num_estimators > 1

    @_maybe_inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:  # Recipe-defined output shape.
        r"""Predict unseen query examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.

        Returns:
            The processed prediction after applying ``recipe.output`` to the
            stacked estimator outputs with shape ``[E, ..., R, *]``.
        """
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

        recipe_execution = cast(
            RecipeExecution,
            self._cache["recipe_execution"],
        )
        with torch.amp.autocast(x.device.type, enabled=False):
            queries = recipe_execution.transform(x, related_tables)

        num_estimators = cast(int, self._cache["num_estimators"])
        estimator_caches = tuple(
            cast(Cache, self._cache[i]) for i in range(num_estimators)
        )
        query_device = queries[0].x.device
        prefetch = (
            num_estimators > 1
            and query_device.type == "cuda"
            and all(query.x.device == query_device for query in queries)
            and self._cache_is_pinned
        )

        transfer_stream = self._cache_transfer_stream
        compute_stream: torch.cuda.Stream | None = None
        next_device_cache: Cache | None = None

        def prefetch_cache(cache: Cache) -> Cache:
            assert transfer_stream is not None
            with torch.cuda.stream(transfer_stream):
                return cache.to(query_device, non_blocking=True)

        outs: list[TableTensor] = []
        try:
            if prefetch:
                if (
                    transfer_stream is None
                    or transfer_stream.device != query_device
                ):
                    if transfer_stream is not None:
                        transfer_stream.synchronize()
                    transfer_stream = torch.cuda.Stream(device=query_device)
                    self._cache_transfer_stream = transfer_stream
                compute_stream = torch.cuda.current_stream(device=query_device)
                next_device_cache = prefetch_cache(estimator_caches[0])

            for i, cache in enumerate(estimator_caches):
                query = queries[i]
                self._validate_query(
                    x_context=cast(TableSchema, cache["x_schema"]),
                    x_query=query.x,
                    related_context_tables=cast(
                        RelatedTablesSchema,
                        cache["related_tables_schema"],
                    ),
                    related_query_tables=query.related_tables,
                )

                if prefetch:
                    assert compute_stream is not None
                    assert transfer_stream is not None
                    assert next_device_cache is not None
                    device_cache = next_device_cache
                    next_device_cache = None
                    compute_stream.wait_stream(transfer_stream)
                    if i + 1 < num_estimators:
                        next_device_cache = prefetch_cache(
                            estimator_caches[i + 1]
                        )
                else:
                    device_cache = cache.to(
                        query.x.device,
                        non_blocking=True,
                    )

                out = self._forward(
                    x_context=None,
                    y_context=None,
                    x_query=query.x,
                    related_context_tables=None,
                    related_query_tables=query.related_tables,
                    cache=device_cache,
                    generator=None,
                    **cast(dict[str, Any], self._cache["kwargs"]),
                )
                if prefetch:
                    assert transfer_stream is not None
                    assert compute_stream is not None
                    # The transfer stream owns these allocations; wait for
                    # compute before releasing them.
                    transfer_stream.wait_stream(compute_stream)
                del device_cache
                out = cast(TableTensor, out.to(query.x.dtype))
                outs.append(out)
        except BaseException as error:
            if prefetch:
                for name, stream in (
                    ("transfer", transfer_stream),
                    ("compute", compute_stream),
                ):
                    if stream is None:
                        continue
                    try:
                        stream.synchronize()
                    except RuntimeError as cleanup_error:
                        error.add_note(
                            f"Failed to synchronize the {name} CUDA stream "
                            f"during cleanup: {cleanup_error}"
                        )
            raise

        # Regression: invert target before stacking estimator outputs.
        if cast(Cache, self._cache[0])["classes"] is None:
            with torch.amp.autocast(x.device.type, enabled=False):
                outs = list(recipe_execution.inverse_transform_target(outs))

        with torch.amp.autocast(x.device.type, enabled=False):
            return recipe_execution.transform_output(outs)

    def clear(self) -> None:
        r"""Clear cached context state created by :meth:`fit`."""
        if self._cache_transfer_stream is not None:
            self._cache_transfer_stream.synchronize()
        self._cache = None
        self._cache_is_pinned = False
        self._cache_transfer_stream = None

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"

    # Abstract Methods ########################################################

    @abstractmethod
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
    @abstractmethod
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
