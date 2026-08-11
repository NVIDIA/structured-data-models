from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, final

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, TableTensor

if TYPE_CHECKING:
    from sdm.models import ICLModel

_ResultT = TypeVar("_ResultT", covariant=True)


class Explainer(ABC, Generic[_ResultT]):  # noqa: D101
    def __init__(self) -> None:
        self._result: _ResultT | None = None

    @property
    def result(self) -> _ResultT:  # noqa: D102
        if self._result is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no explanation result. "
                "Pass it to a model before accessing 'result'."
            )
        return self._result

    @final
    def explain_forward(  # noqa: D102
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
    ) -> TableTensor:
        self._result = None
        prediction, self._result = self._explain_forward(
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
        return prediction

    @final
    def explain_predict(  # noqa: D102
        self,
        model: ICLModel,
        /,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        self._result = None
        prediction, self._result = self._explain_predict(
            model,
            x,
            related_tables,
        )
        return prediction

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
    ) -> tuple[TableTensor, _ResultT]: ...

    @abstractmethod
    def _explain_predict(
        self,
        model: ICLModel,
        /,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> tuple[TableTensor, _ResultT]: ...
