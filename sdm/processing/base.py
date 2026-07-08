import abc
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from sdm import Stype, TableTensor


class Processor(torch.nn.Module, abc.ABC):
    r"""Base processor for tensor-aware table transformations.

    A :class:`Processor` defines a reusable transformation on
    :class:`~sdm.tensor.TableTensor` for feature preprocessing and target
    processing.
    A :class:`Processor` learns any required state via :meth:`fit`, and apply
    the transformation via :meth:`transform`.
    """

    supported_stypes: frozenset[Stype]
    requires_fit: bool

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_supported_stypes(self, input: TableTensor) -> None:
        invalid_stypes: list[str] = []
        for stype, block in input.items():
            if stype == Stype.id or block.size(-1) == 0:
                continue
            if stype not in self.supported_stypes:
                invalid_stypes.append(stype.value)

        if len(invalid_stypes) > 0:
            raise ValueError(
                f"'{self.__class__.__name__}' received non-supported stypes "
                f"{invalid_stypes}"
            )

    def _check_is_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"'{self.__class__.__name__}' is not yet fitted"
            )

    def _fit(self, input: TableTensor) -> None:
        pass

    @abc.abstractmethod
    def _transform(self, input: TableTensor) -> TableTensor:
        pass

    def forward(self, input: TableTensor) -> TableTensor:
        r"""Alias of :meth:`~Processor.transform`.

        Args:
            input: Table to transform.

        Returns:
            Transformed table.
        """
        return self.transform(input)

    def fit(self, input: TableTensor) -> Self:
        r"""Fit the processor.

        Args:
            input: Table used to compute the processor state.
        """
        self._check_supported_stypes(input)
        if self.requires_fit:
            self._fit(input)
            self._fitted = True
        return self

    def transform(self, input: TableTensor) -> TableTensor:
        r"""Transform ``input``.

        Args:
            input: Table to transform.

        Returns:
            Transformed table.
        """
        self._check_supported_stypes(input)
        self._check_is_fitted()
        return self._transform(input)

    def fit_transform(self, input: TableTensor) -> TableTensor:
        r"""Fit the processor and transform ``input``.

        Args:
            input: Table to fit on and transform.

        Returns:
            Transformed table.
        """
        return self.fit(input).transform(input)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    r"""Extends a :class:`Processor` by an inverse transformation."""

    @abc.abstractmethod
    def _inverse_transform(self, input: TableTensor) -> TableTensor: ...

    def inverse_transform(self, input: TableTensor) -> TableTensor:
        r"""Apply the inverse transformation to ``input``.

        Args:
            input: Table in transformed representation.

        Returns:
            Table restored to the representation before :meth:`transform`.
        """
        self._check_is_fitted()
        return self._inverse_transform(input)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
