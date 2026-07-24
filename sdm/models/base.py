import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Processor, Recipe
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

    supported_feature_stypes: ClassVar[frozenset[Stype]]
    supported_target_stypes: ClassVar[frozenset[Stype]]
    supports_related_tables: ClassVar[bool]

    def __init__(self) -> None:
        super().__init__()

        # One cache per ensemble member.
        self._caches: list[Cache] | None = None

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
            recipe: The recipe for pre- and post-processing.
            num_estimators: The number of estimators ``E`` for ensembling.
            generator: Pseudorandom number generator used for sampling during
                pre-processing and model execution.
            kwargs: Additional keyword arguments passed to the model.

        Returns:
            The processed prediction after
            ``recipe_output([E, ..., R_query, *])``.
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
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]

        outs: Sequence[TableTensor] = []
        for recipe in recipes:
            x_context_i = recipe.features.fit_transform(
                x_context,
                generator=generator,
            )
            y_context_i = recipe.target.fit_transform(
                y_context,
                generator=generator,
            )
            x_query_i = recipe.features.transform(x_query)

            related_context_tables_i = related_query_tables_i = None
            if related_context_tables is not None:
                related_processors = {
                    table_name: copy.deepcopy(recipe.features)
                    for table_name in related_context_tables.tables
                }
                related_context_tables_i = replace(
                    related_context_tables,
                    tables={
                        name: related_processors[name].fit_transform(
                            t,
                            generator=generator,
                        )
                        for name, t in related_context_tables.tables.items()
                    },
                )
                assert related_query_tables is not None
                related_query_tables_i = replace(
                    related_query_tables,
                    tables={
                        name: related_processors[name].transform(t)
                        for name, t in related_query_tables.tables.items()
                    },
                )

            self._validate_context(
                x=x_context_i,
                y=y_context_i,
                related_tables=related_context_tables_i,
            )
            self._validate_query(
                x_context=x_context_i.schema,
                x_query=x_query_i,
                related_context_tables=related_context_tables_i.schema
                if related_context_tables_i is not None
                else None,
                related_query_tables=related_query_tables_i,
            )

            out = self._forward(
                x_context=x_context_i,
                y_context=y_context_i,
                x_query=x_query_i,
                related_context_tables=related_context_tables_i,
                related_query_tables=related_query_tables_i,
                cache=None,
                generator=generator,
                **kwargs,
            )
            if y_context_i.numerical.size(-1) == 1:
                if not isinstance(recipe.target, InvertibleMixin):
                    raise RuntimeError("Target recipe is not invertible")
                out = recipe.target.inverse_transform(out)
            outs.append(out)

        out: TableTensor = cast(TableTensor, torch.stack(outs, dim=0))
        out = cast(TableTensor, out.to(x_query_i.dtype))
        return recipe.output.transform(out)

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
            recipe: The recipe for pre- and post-processing. If ``None``, no
                recipe is applied.
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

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]

        self.clear()
        caches: list[Cache] = []
        for recipe in recipes:
            x_i = recipe.features.fit_transform(x, generator=generator)
            y_i = recipe.target.fit_transform(y, generator=generator)

            related_tables_i = None
            related_processors = None
            if related_tables is not None:
                related_processors = {
                    table_name: copy.deepcopy(recipe.features)
                    for table_name in related_tables.tables
                }
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: related_processors[name].fit_transform(
                            t,
                            generator=generator,
                        )
                        for name, t in related_tables.tables.items()
                    },
                )

            self._validate_context(
                x=x_i,
                y=y_i,
                related_tables=related_tables_i,
            )

            cache = Cache(
                recipe=recipe,
                x_schema=x_i.schema,
                related_processors=related_processors,
                related_tables_schema=related_tables_i.schema
                if related_tables_i is not None
                else None,
                classes=y_i.categorical.categories[0]
                if y_i.categorical.size(-1) > 0
                else None,
                kwargs=kwargs,
            )

            self._forward(
                x_context=x_i,
                y_context=y_i,
                x_query=None,
                related_context_tables=related_tables_i,
                related_query_tables=None,
                cache=cache,
                generator=generator,
                **kwargs,
            )
            cache = cache.cpu().freeze()
            caches.append(cache)
        self._caches = caches

    def clear(self) -> None:
        r"""Clears cached in-context examples and the fitted recipe."""
        self._caches = None

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
        Returns:
            The processed prediction after
            ``recipe_output([E, ..., R, *])``.
        """
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

        if self._caches is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
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

        outs: Sequence[TableTensor] = []
        for cache in self._caches:
            recipe = cast(Recipe, cache["recipe"])
            x_i = recipe.features.transform(x)

            related_tables_i = None
            if related_tables is not None:
                related_processors = cast(
                    Mapping[str, Processor],
                    cache["related_processors"],
                )
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: related_processors[name].transform(t)
                        for name, t in related_tables.tables.items()
                    },
                )

            self._validate_query(
                x_context=cast(TableSchema, cache["x_schema"]),
                x_query=x_i,
                related_context_tables=cast(
                    RelatedTablesSchema,
                    cache["related_tables_schema"],
                ),
                related_query_tables=related_tables_i,
            )

            out = self._forward(
                x_context=None,
                y_context=None,
                x_query=x_i,
                related_context_tables=None,
                related_query_tables=related_tables_i,
                cache=cache.to(x_i.device),
                generator=None,
                **cast(dict[str, Any], cache["kwargs"]),
            )
            if cache["classes"] is None:
                if not isinstance(recipe.target, InvertibleMixin):
                    raise RuntimeError("Target recipe is not invertible")
                out = recipe.target.inverse_transform(out)
            outs.append(out)

        out: TableTensor = cast(TableTensor, torch.stack(outs, dim=0))
        out = cast(TableTensor, out.to(x_i.dtype))
        return recipe.output.transform(out)

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
            raise ValueError(
                f"'{self.__class__.__name__}' received unsupported feature "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )
        invalid = y.active_stypes - self.supported_target_stypes
        if len(invalid) > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' received unsupported target "
                f"stypes: {', '.join(stype.value for stype in invalid)}"
            )

        if related_tables is not None:
            if not self.supports_related_tables:
                raise ValueError(
                    f"'{self.__class__.__name__}' does not support related "
                    f"tables"
                )
            for table_name, table in related_tables.tables.items():
                invalid = table.active_stypes - self.supported_feature_stypes
                invalid = invalid - {Stype.id}
                if len(invalid) > 0:
                    raise ValueError(
                        f"'{self.__class__.__name__}' received unsupported "
                        f"feature stypes in related table '{table_name}': "
                        f"{', '.join(stype.value for stype in invalid)}"
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
