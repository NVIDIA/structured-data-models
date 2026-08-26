import abc
from typing import Any, Generic, TypeVar, cast, final, overload

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor
from sdm.models import ICLModel

T = TypeVar("T", covariant=True)


class ICLExplainer(abc.ABC, Generic[T]):  # noqa: D101
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
    ) -> T: ...

    @overload
    def explain(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
    ) -> T: ...

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
    ) -> T:

        num_estimators = kwargs.get("num_estimators", 1)
        if num_estimators != 1:
            raise RuntimeError(
                f"{model.__class__.__name__!r} only supports explaining "
                f"models with a single estimator (got {num_estimators})"
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
                generator=generator,
                **kwargs,
            )

        if model._cache is None:
            raise RuntimeError(
                f"{model.__class__.__name__!r} is not fitted. Pass required "
                f"context to 'explain()' or call "
                f"'{model.__class__.__name__}.fit()' first."
            )

        _, cache = model._cache._state()
        num_estimators = cast(int, cache["num_estimators"])
        if num_estimators != 1:
            raise RuntimeError(
                f"{model.__class__.__name__!r} only supports explaining "
                f"models fitted with a single estimator "
                f"(got {num_estimators})"
            )

        return self._explain_predict(
            model,
            x_query,
            related_query_tables,
        )

    @abc.abstractmethod
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
    ) -> T: ...

    @abc.abstractmethod
    def _explain_predict(
        self,
        model: ICLModel,
        x_query: Tensor | TableTensor,
        related_query_tables: RelatedTables | None = None,
    ) -> T: ...
