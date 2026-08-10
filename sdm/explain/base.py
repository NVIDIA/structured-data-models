from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, final

from sdm import TableTensor

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
    def explain(
        self,
        operation: Callable[..., TableTensor],
        prediction: TableTensor,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        """Explain a model operation and return its prediction.

        Args:
            operation: Model operation that produced ``prediction``.
            prediction: Prediction returned by ``operation``.
            *args: Positional arguments passed to ``operation``.
            **kwargs: Keyword arguments passed to ``operation``.

        Returns:
            ``prediction`` unchanged.
        """
        self._result = None
        self._result = self._explain(
            operation,
            prediction,
            *args,
            **kwargs,
        )
        return prediction

    @abstractmethod
    def _explain(
        self,
        operation: Callable[..., TableTensor],
        prediction: TableTensor,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _ResultT: ...
