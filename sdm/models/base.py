import contextlib
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar, cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Recipe, RecipeContext


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

        if num_estimators <= 0:
            raise ValueError("num_estimators must be positive")

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]
        original_class_labels = self._class_labels(y_context)

        outs: list[TableTensor] = []
        contexts: list[RecipeContext] = []
        for estimator_index, member_recipe in enumerate(recipes):
            # TODO Transform related tables.
            y_context_i = member_recipe.target.fit_transform(y_context)
            out = self._forward(
                x_context=member_recipe.features.fit_transform(x_context),
                y_context=y_context_i,
                x_query=member_recipe.features.transform(x_query),
                related_context_tables=None,
                related_query_tables=None,
                cache=None,
            )
            contexts.append(
                self._recipe_context(
                    estimator_index=estimator_index,
                    target=y_context_i,
                    recipe=member_recipe,
                    class_indices=self._class_indices(
                        original_class_labels,
                        self._class_labels(y_context_i),
                    ),
                )
            )
            outs.append(out)

        # [..., estimators, rows, outputs]
        out = cast(
            TableTensor,
            torch.stack(cast(list[Tensor], outs), dim=-3),
        )
        return recipes[0].output.transform(out, context=contexts)

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

        if num_estimators <= 0:
            raise ValueError("num_estimators must be positive")

        recipe = self.default_recipe() if recipe is None else recipe
        recipes = [copy.deepcopy(recipe) for _ in range(num_estimators)]
        original_class_labels = self._class_labels(y)

        self.clear()
        caches: list[Cache] = []
        for estimator_index, member_recipe in enumerate(recipes):
            y_i = member_recipe.target.fit_transform(y)
            context = self._recipe_context(
                estimator_index=estimator_index,
                target=y_i,
                recipe=member_recipe,
                class_indices=self._class_indices(
                    original_class_labels,
                    self._class_labels(y_i),
                ),
            )
            cache = Cache(
                recipe=member_recipe,
                context=context,
                classes=y_i.categorical.categories[0]
                if y_i.categorical.size(-1) > 0
                else None,
            )
            self._forward(
                x_context=member_recipe.features.fit_transform(x),
                y_context=y_i,
                x_query=None,
                related_context_tables=None,
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

        outs: list[TableTensor] = []
        contexts: list[RecipeContext] = []
        for cache in self._caches:
            member_recipe = cast(Recipe, cache["recipe"])
            out = self._forward(
                x_context=None,
                y_context=None,
                x_query=member_recipe.features.transform(x),
                related_context_tables=None,
                related_query_tables=None,
                cache=cache,
            )
            contexts.append(cast(RecipeContext, cache["context"]))
            outs.append(out)

        # [..., estimators, rows, outputs]
        out = cast(
            TableTensor,
            torch.stack(cast(list[Tensor], outs), dim=-3),
        )
        output_recipe = cast(Recipe, self._caches[0]["recipe"])
        return output_recipe.output.transform(out, context=contexts)

    # Helpers #################################################################

    @staticmethod
    def _recipe_context(
        *,
        estimator_index: int,
        target: TableTensor,
        recipe: Recipe,
        class_indices: tuple[int, ...] | None,
    ) -> RecipeContext:
        if target.numerical.size(-1) == 1:
            if not isinstance(recipe.target, InvertibleMixin):
                raise ValueError(
                    "Expected Recipe target processing to support "
                    "inverse_transform for regression."
                )
            return RecipeContext(
                estimator_index=estimator_index,
                task="regression",
                target_inverse=recipe.target,
            )
        if target.categorical.size(-1) == 1:
            return RecipeContext(
                estimator_index=estimator_index,
                task="classification",
                class_indices=class_indices,
            )
        raise ValueError(
            "Expected the transformed target to contain one numerical or "
            "categorical column."
        )

    @staticmethod
    def _class_indices(
        original_class_labels: tuple[str, ...] | None,
        member_class_labels: tuple[str, ...] | None,
    ) -> tuple[int, ...] | None:
        if member_class_labels is None:
            return None
        if original_class_labels is None:
            raise ValueError(
                "Cannot map estimator classes without original class labels."
            )

        member_indices = {
            label: index for index, label in enumerate(member_class_labels)
        }
        try:
            return tuple(
                member_indices[label] for label in original_class_labels
            )
        except KeyError as exc:
            raise ValueError(
                f"Estimator output is missing original class {exc.args[0]!r}."
            ) from exc

    @staticmethod
    def _class_labels(target: TableTensor) -> tuple[str, ...] | None:
        if target.categorical.size(-1) != 1:
            return None

        values = target.categorical.categories[0].tolist()
        labels = tuple(str(value) for value in values)
        if len(labels) != len(set(labels)):
            raise ValueError(
                "Expected categorical targets to have unique class-label "
                "representations."
            )
        return labels

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
