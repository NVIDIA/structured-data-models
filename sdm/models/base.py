import contextlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, TableTensor
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Recipe
from sdm.processing._recipe_execution import _MemberContext, _RecipeExecution
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


def _target_classes(y: TableTensor) -> Tensor | None:
    if y.categorical.size(-1) > 0:
        return y.categorical.categories[0]
    return None


def _is_regression(contexts: tuple[_MemberContext, ...]) -> bool:
    return all(_target_classes(context.y) is None for context in contexts)


def _postprocess_outputs(
    execution: _RecipeExecution,
    outputs: list[TableTensor],
    *,
    device_type: str,
) -> TableTensor:
    with torch.amp.autocast(device_type, enabled=False):
        if _is_regression(execution.contexts):
            if not isinstance(execution.recipe.target, InvertibleMixin):
                raise RuntimeError("Target recipe is not invertible")
            outputs = list(execution.inverse_transform_target(outputs))
        return execution.transform_output(outputs)


class ICLModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`ICLModel` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    #: Semantic types supported for input feature columns in this model.
    supported_feature_stypes: ClassVar[frozenset[Stype]]

    #: Semantic types supported for target columns in this model.
    supported_target_stypes: ClassVar[frozenset[Stype]]

    #: Whether this model supports additional related context.
    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None
        # One execution for "vectorized" (spanning all estimators), or one
        # per estimator for "sequential".
        self._recipe_executions: tuple[_RecipeExecution, ...] | None = None

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
        recipe_execution: Literal["sequential", "vectorized"] = "vectorized",
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
            recipe: The recipe for pre- and post-processing.
            num_estimators: The number of estimators ``E`` for ensembling.
            recipe_execution: How to run the recipe across estimators.
                ``"sequential"`` processes one estimator at a time (less
                memory). ``"vectorized"`` processes all estimators in one
                batched pass (faster, more GPU memory; fall back to
                ``"sequential"`` on out-of-memory errors). The model
                itself always runs once per estimator either way.
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

        recipe = self.default_recipe() if recipe is None else recipe

        if recipe_execution == "vectorized":
            with torch.amp.autocast(x_query.device.type, enabled=False):
                execution = recipe.bind(
                    x_context=x_context,
                    y_context=y_context,
                    related_context_tables=related_context_tables,
                    num_members=num_estimators,
                    generator=generator,
                )
                queries = execution.transform(
                    x_query=x_query,
                    related_query_tables=related_query_tables,
                )

            outs: list[TableTensor] = []
            for context, query in zip(
                execution.contexts,
                queries,
                strict=True,
            ):
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

            return _postprocess_outputs(
                execution,
                outs,
                device_type=x_query.device.type,
            )
        if recipe_execution != "sequential":
            raise AssertionError(
                f"Unexpected recipe_execution {recipe_execution!r}"
            )

        outs = []
        last_execution: _RecipeExecution | None = None
        for _ in range(num_estimators):
            with torch.amp.autocast(x_query.device.type, enabled=False):
                execution = recipe.bind(
                    x_context=x_context,
                    y_context=y_context,
                    related_context_tables=related_context_tables,
                    num_members=1,
                    generator=generator,
                )
                (query,) = execution.transform(
                    x_query=x_query,
                    related_query_tables=related_query_tables,
                )
            (context,) = execution.contexts

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
            if _is_regression(execution.contexts):
                if not isinstance(execution.recipe.target, InvertibleMixin):
                    raise RuntimeError("Target recipe is not invertible")
                with torch.amp.autocast(x_query.device.type, enabled=False):
                    (out,) = execution.inverse_transform_target([out])
            outs.append(out)
            last_execution = execution

        assert last_execution is not None
        out = cast(
            TableTensor,
            torch.stack(cast(list[Tensor], outs), dim=0),
        )
        with torch.amp.autocast(x_query.device.type, enabled=False):
            return last_execution.recipe.output.transform(out)

    @_maybe_inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        recipe_execution: Literal["sequential", "vectorized"] = "vectorized",
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
            recipe: The recipe for pre- and post-processing. If ``None``, no
                recipe is applied.
            num_estimators: The number of estimators for ensembling.
            recipe_execution: How to run the recipe across estimators.
                ``"sequential"`` processes one estimator at a time (less
                memory). ``"vectorized"`` processes all estimators in one
                batched pass (faster, more GPU memory; fall back to
                ``"sequential"`` on out-of-memory errors). The model
                itself always runs once per estimator either way.
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

        recipe = self.default_recipe() if recipe is None else recipe

        self.clear()
        if recipe_execution == "vectorized":
            with torch.amp.autocast(x.device.type, enabled=False):
                executions = (
                    recipe.bind(
                        x_context=x,
                        y_context=y,
                        related_context_tables=related_tables,
                        num_members=num_estimators,
                        generator=generator,
                    ),
                )
        elif recipe_execution == "sequential":
            sequential_executions: list[_RecipeExecution] = []
            for _ in range(num_estimators):
                with torch.amp.autocast(x.device.type, enabled=False):
                    sequential_executions.append(
                        recipe.bind(
                            x_context=x,
                            y_context=y,
                            related_context_tables=related_tables,
                            num_members=1,
                            generator=generator,
                        )
                    )
            executions = tuple(sequential_executions)
        else:
            raise AssertionError(
                f"Unexpected recipe_execution {recipe_execution!r}"
            )

        caches: list[Cache] = []
        for execution in executions:
            for context in execution.contexts:
                self._validate_context(
                    x=context.x,
                    y=context.y,
                    related_tables=context.related_tables,
                )

                cache = Cache(
                    x_schema=context.x.schema,
                    related_tables_schema=context.related_tables.schema
                    if context.related_tables is not None
                    else None,
                    classes=_target_classes(context.y),
                    kwargs=kwargs,
                )

                self._forward(
                    x_context=context.x,
                    y_context=context.y,
                    x_query=None,
                    related_context_tables=context.related_tables,
                    related_query_tables=None,
                    cache=cache,
                    generator=generator,
                    **kwargs,
                )
                if num_estimators > 1:
                    cache = cache.cpu()
                cache = cache.freeze()
                caches.append(cache)
        self._caches = caches
        self._recipe_executions = executions

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

        if self._caches is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        if related_tables is not None:
            if self._caches[0]["related_tables_schema"] is None:
                raise ValueError(
                    "Expected related tables to be provided together"
                )
            related_tables = related_tables.select_tables(
                tables=cast(
                    RelatedTablesSchema,
                    self._caches[0]["related_tables_schema"],
                ).tables,
            )

        assert self._recipe_executions is not None

        queries = []
        with torch.amp.autocast(x.device.type, enabled=False):
            for execution in self._recipe_executions:
                queries.extend(
                    execution.transform(
                        x_query=x,
                        related_query_tables=related_tables,
                    )
                )

        if len(self._recipe_executions) == 1:
            (execution,) = self._recipe_executions
            outs: list[TableTensor] = []
            for cache, query in zip(
                self._caches,
                queries,
                strict=True,
            ):
                self._validate_query(
                    x_context=cast(TableSchema, cache["x_schema"]),
                    x_query=query.x,
                    related_context_tables=cast(
                        RelatedTablesSchema,
                        cache["related_tables_schema"],
                    ),
                    related_query_tables=query.related_tables,
                )

                out = self._forward(
                    x_context=None,
                    y_context=None,
                    x_query=query.x,
                    related_context_tables=None,
                    related_query_tables=query.related_tables,
                    cache=cache.to(query.x.device),
                    generator=None,
                    **cast(dict[str, Any], cache["kwargs"]),
                )
                out = cast(TableTensor, out.to(query.x.dtype))
                outs.append(out)

            return _postprocess_outputs(
                execution,
                outs,
                device_type=x.device.type,
            )

        outs = []
        last_execution: _RecipeExecution | None = None
        for execution, cache, query in zip(
            self._recipe_executions,
            self._caches,
            queries,
            strict=True,
        ):
            self._validate_query(
                x_context=cast(TableSchema, cache["x_schema"]),
                x_query=query.x,
                related_context_tables=cast(
                    RelatedTablesSchema,
                    cache["related_tables_schema"],
                ),
                related_query_tables=query.related_tables,
            )

            out = self._forward(
                x_context=None,
                y_context=None,
                x_query=query.x,
                related_context_tables=None,
                related_query_tables=query.related_tables,
                cache=cache.to(query.x.device),
                generator=None,
                **cast(dict[str, Any], cache["kwargs"]),
            )
            out = cast(TableTensor, out.to(query.x.dtype))
            if _is_regression(execution.contexts):
                if not isinstance(execution.recipe.target, InvertibleMixin):
                    raise RuntimeError("Target recipe is not invertible")
                with torch.amp.autocast(x.device.type, enabled=False):
                    (out,) = execution.inverse_transform_target([out])
            outs.append(out)
            last_execution = execution

        assert last_execution is not None
        out = cast(
            TableTensor,
            torch.stack(cast(list[Tensor], outs), dim=0),
        )
        with torch.amp.autocast(x.device.type, enabled=False):
            return last_execution.recipe.output.transform(out)

    def clear(self) -> None:
        r"""Clear cached context state created by :meth:`fit`."""
        self._caches = None
        self._recipe_executions = None

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
            stypes = ", ".join(f"{stype.value!r}" for stype in invalid)
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
            stypes = ", ".join(f"{stype.value!r}" for stype in invalid)
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
                    stypes = ", ".join(f"{stype.value!r}" for stype in invalid)
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
