import abc
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from sdm.tensor import TableTensor


class Processor(torch.nn.Module, abc.ABC):
    """Fittable, table-in/table-out transform.

    Subclass and implement ``_transform`` (the transform operation). Override
    ``_fit`` to learn state from a :class:`TableTensor` (the default is a
    no-op). For an inverse, also mix in :class:`InvertibleMixin` and implement
    ``_inverse_transform``. Set ``requires_fit = False`` for stateless
    processors that can safely run without a prior ``fit`` call.
    """

    requires_fit: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_is_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"'{self.__class__.__name__}' is not fitted; "
                "call 'fit()' before."
            )

    def _fit(self, input: TableTensor) -> None:
        pass

    @abc.abstractmethod
    def _transform(self, input: TableTensor) -> TableTensor:
        pass

    def forward(self, input: TableTensor) -> TableTensor:
        """Alias of :meth:`~Processor.transform`.

        This is the :class:`torch.nn.Module` entry point, so
        ``processor(input)`` and ``processor.transform(input)`` share the same
        fitted-state checks.

        Args:
            input: Table to transform.

        Returns:
            Transformed table.
        """
        return self.transform(input)

    def fit(self, input: TableTensor) -> Self:
        """Fit the processor on ``input`` and return it.

        Args:
            input: Feature table used to compute the processor state.

        Returns:
            This processor.
        """
        if self.requires_fit:
            self._fit(input)
            self._fitted = True
        return self

    def transform(self, input: TableTensor) -> TableTensor:
        """Transform ``input`` using the fitted processor.

        Args:
            input: Table to transform.

        Returns:
            Transformed table.
        """
        self._check_is_fitted()
        return self._transform(input)

    def fit_transform(self, input: TableTensor) -> TableTensor:
        """Fit on ``input`` and return the transformed result.

        Args:
            input: Feature table to fit on and transform.

        Returns:
            Transformed table.
        """
        return self.fit(input).transform(input)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    """Adds ``inverse_transform`` to a :class:`Processor`.

    Combine with :class:`Processor` and implement ``_inverse_transform``,
    e.g. ``class StandardScale(Processor, InvertibleMixin): ...``.
    """

    @abc.abstractmethod
    def _inverse_transform(self, input: TableTensor) -> TableTensor: ...

    def inverse_transform(self, input: TableTensor) -> TableTensor:
        """Invert the transform of ``input`` using the fitted processor.

        Args:
            input: Table in transformed space.

        Returns:
            Table mapped back to the original processor space.
        """
        self._check_is_fitted()
        return self._inverse_transform(input)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
