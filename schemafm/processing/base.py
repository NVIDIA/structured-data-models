from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeVar

import torch
from torch import Tensor

SelfProcessor = TypeVar("SelfProcessor", bound="Processor")


class Processor(torch.nn.Module, abc.ABC):
    """Fittable, tensor-in/tensor-out transform.

    Subclass and implement ``_transform``. Override ``_fit`` to learn state
    from data (the default is a no-op). For an inverse, also mix in
    ``InvertibleMixin`` and implement ``_inverse_transform``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"'{self.__class__.__name__}' is not fitted; "
                "call 'fit' before 'transform' or 'inverse_transform'."
            )

    def _fit(self, input: Tensor) -> None:
        pass

    @abc.abstractmethod
    def _transform(self, input: Tensor) -> Tensor: ...

    def fit(self: SelfProcessor, input: Tensor) -> SelfProcessor:
        self._fit(input)
        self._fitted = True
        return self

    def transform(self, input: Tensor) -> Tensor:
        self._check_is_fitted()
        return self._transform(input)

    def fit_transform(self, input: Tensor) -> Tensor:
        return self.fit(input).transform(input)


class InvertibleMixin(abc.ABC):
    """Adds ``inverse_transform`` to a :class:`Processor`.

    Combine with :class:`Processor` and implement ``_inverse_transform``,
    e.g. ``class StandardScale(Processor, InvertibleMixin): ...``.
    """

    @abc.abstractmethod
    def _inverse_transform(self, input: Tensor) -> Tensor: ...

    def inverse_transform(self, input: Tensor) -> Tensor:
        self._check_is_fitted()
        return self._inverse_transform(input)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
