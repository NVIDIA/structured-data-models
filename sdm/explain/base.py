from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor
from sdm.cache import Cache

if TYPE_CHECKING:
    from sdm.models import ICLModel

_ResultT = TypeVar("_ResultT", covariant=True)


class Explainer(ABC, Generic[_ResultT]):  # noqa: D101
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
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _ResultT:
        if (x_context is None) != (y_context is None):
            raise ValueError(
                "'x_context' and 'y_context' must be provided together"
            )

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
                num_estimators=num_estimators,
                generator=generator,
                **kwargs,
            )

        if (
            related_context_tables is not None
            or recipe is not None
            or num_estimators != 1
            or generator is not None
            or kwargs
        ):
            raise TypeError(
                "Forward arguments require 'x_context' and 'y_context'"
            )
        cache = model._cache
        if cache is None:
            raise RuntimeError(
                f"{model.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{model.__class__.__name__}.fit()' before."
            )
        return self._explain_predict(
            model,
            x_query,
            related_query_tables,
            cache=cache,
        )

    @abstractmethod
    def _explain_forward(
        self,
        model: ICLModel,
        /,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        related_context_tables: RelatedTables | None = None,
        related_query_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> _ResultT: ...

    @abstractmethod
    def _explain_predict(
        self,
        model: ICLModel,
        /,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
        *,
        cache: Cache,
    ) -> _ResultT: ...
