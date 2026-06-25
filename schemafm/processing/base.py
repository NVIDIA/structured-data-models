from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor
from typing_extensions import Self


class Processor(torch.nn.Module, abc.ABC):
    """Fittable, tensor-in/tensor-out transform.

    Subclass and implement ``forward`` (the transform). Override ``_fit`` to
    learn state from data (the default is a no-op). For an inverse, also mix
    in ``InvertibleMixin`` and implement ``_inverse_transform``.

    Set ``requires_fit = False`` for stateless processors that can safely run
    without a prior ``fit`` call.

    Processors assume the caller has selected compatible tensor blocks and
    normalized dtypes before calling ``fit`` or ``transform``. Table-level
    dispatch and casting will be formalized separately from this base class.
    """

    requires_fit: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_is_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"'{self.__class__.__name__}' is not fitted; "
                "call 'fit()' before."
            )

    def _fit(self, input: Tensor) -> None:
        pass

    @abc.abstractmethod
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Transform ``input`` and return the result.

        Use :meth:`transform` to run with a fitted-state check. Extra
        arguments carry explicit context for stateless processors without
        storing it on the processor.
        """

    def fit(self, input: Tensor) -> Self:
        """Fit the processor on ``input`` and return it."""
        self._fit(input)
        self._fitted = True
        return self

    def transform(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Transform ``input`` using the fitted processor."""
        self._check_is_fitted()
        return self(input, *args, **kwargs)

    def fit_transform(self, input: Tensor) -> Tensor:
        """Fit on ``input`` and return the transformed result."""
        return self.fit(input).transform(input)


class InvertibleMixin(abc.ABC):
    """Adds ``inverse_transform`` to a :class:`Processor`.

    Combine with :class:`Processor` and implement ``_inverse_transform``,
    e.g. ``class StandardScale(Processor, InvertibleMixin): ...``.
    """

    @abc.abstractmethod
    def _inverse_transform(self, input: Tensor) -> Tensor: ...

    def inverse_transform(self, input: Tensor) -> Tensor:
        """Invert the transform of ``input`` using the fitted processor."""
        self._check_is_fitted()
        return self._inverse_transform(input)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
