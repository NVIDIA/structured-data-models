from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final, overload

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor

if TYPE_CHECKING:
    from sdm.models import ICLModel

_ResultT = TypeVar("_ResultT", covariant=True)


class Explainer(ABC, Generic[_ResultT]):  # noqa: D101
    @overload
    def explain(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
        *,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _ResultT: ...

    @overload
    def explain(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
    ) -> _ResultT: ...

    @final
    def explain(  # noqa: D102
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
        *,
        x_context: Tensor | TableTensor | None = None,
        y_context: Tensor | TableTensor | None = None,
        related_context_tables: RelatedTables | None = None,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _ResultT:
        if x_context is not None:
            assert y_context is not None
            return self._explain_forward(
                model,
                x_context,
                y_context,
                x_query,
                related_context_tables,
                related_query_tables,
                recipe=recipe,
                generator=generator,
                **kwargs,
            )

        if model._cache is None:
            raise RuntimeError(
                f"{model.__class__.__name__!r} has no cache. Pass 'x_context' "
                "and 'y_context', and any required related tables to "
                f"'explain()', or call '{model.__class__.__name__}.fit()' "
                "first."
            )

        return self._explain_predict(
            model,
            x_query,
            related_query_tables,
        )

    @abstractmethod
    def _explain_forward(
        self,
        model: ICLModel,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _ResultT: ...

    @abstractmethod
    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
    ) -> _ResultT: ...
