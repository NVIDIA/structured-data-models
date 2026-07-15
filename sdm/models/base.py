import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import ClassVar, cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Processor, Recipe


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


class Model(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`Model` defines the public interface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

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
    ) -> TableTensor:  # [..., R_query, *]
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
            num_estimators: The number of estimators for ensembling.

        Returns:
            The prediction ``[..., R_query, *]`` for all query examples.
        """
        # TODO Add validation.
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(y_context, TableTensor):
            y_context = TableTensor.from_tensor(y_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]

        outs: Sequence[TableTensor] = []
        for recipe in recipes:
            y_context_i = recipe.target.fit_transform(y_context)
            x_context_i = recipe.features.fit_transform(x_context)

            related_context_tables_i = related_query_tables_i = None
            if related_context_tables is not None:
                related_processors = {
                    table_name: copy.deepcopy(recipe.features)
                    for table_name in related_context_tables.tables
                }
                related_context_tables_i = replace(
                    related_context_tables,
                    tables={
                        name: related_processors[name].fit_transform(t)
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

            out = self._forward(
                x_context=x_context_i,
                y_context=y_context_i,
                x_query=recipe.features.transform(x_query),
                related_context_tables=related_context_tables_i,
                related_query_tables=related_query_tables_i,
                cache=None,
            )
            if y_context_i.numerical.size(-1) == 1:
                assert isinstance(recipe.target, InvertibleMixin)
                out = recipe.target.inverse_transform(out)
            outs.append(out)

        out: TableTensor = cast(TableTensor, torch.stack(outs, dim=0))
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
        """
        # TODO Add validation.
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        if not isinstance(y, TableTensor):
            y = TableTensor.from_tensor(y)

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]

        self.clear()
        caches: list[Cache] = []
        for recipe in recipes:
            y_i = recipe.target.fit_transform(y)

            cache = Cache(
                recipe=recipe,
                classes=y_i.categorical.categories[0]
                if y_i.categorical.size(-1) > 0
                else None,
            )

            related_tables_i = None
            if related_tables is not None:
                related_processors = cache["related_processors"] = {
                    table_name: copy.deepcopy(recipe.features)
                    for table_name in related_tables.tables
                }
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: related_processors[name].fit_transform(t)
                        for name, t in related_tables.tables.items()
                    },
                )

            self._forward(
                x_context=recipe.features.fit_transform(x),
                y_context=y_i,
                x_query=None,
                related_context_tables=related_tables_i,
                related_query_tables=None,
                cache=cache,
            )
            cache.freeze()
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
    ) -> TableTensor:  # [..., R, *]
        r"""Predict unseen query examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor of query examples with shape
                ``[..., R, D]`` with ``R`` rows and ``D`` columns.
            related_tables: Related context for query examples.

        Returns:
            The prediction ``[..., R, *]`` for all query examples.
        """
        # TODO Add validation.
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

        if self._caches is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        outs: Sequence[TableTensor] = []
        for cache in self._caches:
            recipe = cast(Recipe, cache["recipe"])

            related_tables_i = None
            if related_tables is not None:
                related_processors = cast(
                    Mapping[str, Processor],
                    cache["related_feature_processors"],
                )
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: related_processors[name].transform(t)
                        for name, t in related_tables.tables.items()
                    },
                )

            out = self._forward(
                x_context=None,
                y_context=None,
                x_query=recipe.features.transform(x),
                related_context_tables=None,
                related_query_tables=related_tables_i,
                cache=cache,
            )
            if cache["classes"] is None:
                assert isinstance(recipe.target, InvertibleMixin)
                out = recipe.target.inverse_transform(out)
            outs.append(out)

        out: TableTensor = cast(TableTensor, torch.stack(outs, dim=0))
        return recipe.output.transform(out)

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
    ) -> TableTensor:  # [..., R_query, *]
        pass

    @classmethod
    @abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""
