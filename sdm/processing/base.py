import abc
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from torch import Tensor
from typing_extensions import Self


class Processor(torch.nn.Module, abc.ABC):
    """Fittable, tensor-in/tensor-out transform.

    Subclass and implement ``_transform`` (the transform operation). Override
    ``_fit`` to learn state from data (the default is a no-op). For an inverse,
    also mix in :class:`InvertibleMixin` and implement ``_inverse_transform``.

    ``Sequential`` routes steps with ``input_scope = "block"`` to one tensor
    block selected by the sequence. Steps with ``input_scope = "table"``
    receive and return the whole table. Set ``requires_fit = False`` for
    stateless processors that can safely run without a prior ``fit`` call.
    """

    requires_fit: bool = True
    input_scope: ClassVar[Literal["block", "table"]] = "block"

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
    def _transform(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Alias of :meth:`~Processor.transform`.

        This is the :class:`torch.nn.Module` entry point, so
        ``processor(input)`` and ``processor.transform(input)`` share the same
        fitted-state checks.

        Args:
            input: Tensor to transform. Concrete processors document the
                accepted shape.

        Returns:
            Tensor with the shape documented by the concrete processor.
        """
        return self.transform(input)

    def fit(self, input: Tensor) -> Self:
        """Fit the processor on ``input`` and return it.

        Args:
            input: Feature tensor used to compute the processor state.
                Concrete processors document the accepted shape.

        Returns:
            This processor.
        """
        if self.requires_fit:
            self._fit(input)
            self._fitted = True
        return self

    def transform(self, input: Tensor) -> Tensor:
        """Transform ``input`` using the fitted processor.

        Args:
            input: Tensor to transform. Concrete processors document the
                accepted shape.

        Returns:
            Transformed tensor with the shape documented by the concrete
            processor.
        """
        self._check_is_fitted()
        return self._transform(input)

    def fit_transform(self, input: Tensor) -> Tensor:
        """Fit on ``input`` and return the transformed result.

        Args:
            input: Feature tensor to fit on and transform. Concrete
                processors document the accepted shape.

        Returns:
            Transformed tensor with the shape documented by the concrete
            processor.
        """
        return self.fit(input).transform(input)

    def resolve(
        self,
        *,
        estimator: int = 0,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Return the concrete processor for a processing context."""
        return self

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    """Adds ``inverse_transform`` to a :class:`Processor`.

    Combine with :class:`Processor` and implement ``_inverse_transform``,
    e.g. ``class StandardScale(Processor, InvertibleMixin): ...``.
    """

    @abc.abstractmethod
    def _inverse_transform(self, input: Tensor) -> Tensor: ...

    def inverse_transform(self, input: Tensor) -> Tensor:
        """Invert the transform of ``input`` using the fitted processor.

        Args:
            input: Tensor in transformed space. Concrete processors
                document the accepted shape.

        Returns:
            Tensor mapped back to the original processor space.
        """
        self._check_is_fitted()
        return self._inverse_transform(input)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
