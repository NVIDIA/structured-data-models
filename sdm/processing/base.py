from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self, TypeAlias

import torch

from sdm import Stype, TableTensor

if TYPE_CHECKING:
    from sdm.processing import Sequential

OperatesOnStypes: TypeAlias = frozenset[Stype]
UnoperatedStypePolicy: TypeAlias = Literal["preserve", "error", "opaque"]


class Processor(torch.nn.Module, abc.ABC):
    r"""Base processor for tensor-aware table transformations.

    A :class:`Processor` defines a reusable transformation on
    :class:`~sdm.tensor.TableTensor` for feature, target and output
    preprocessing.
    A :class:`Processor` learns any required state via :meth:`fit`, and applies
    the transformation via :meth:`transform`. Implementations preserve the row
    and batch dimensions. Batch dimensions are processed independently.

    Processors declare the semantic column types they operate on via
    :attr:`operates_on_stypes`. Active non-operated semantic types follow
    :attr:`unoperated_stype_policy`.
    """

    operates_on_stypes: OperatesOnStypes
    unoperated_stype_policy: UnoperatedStypePolicy = "preserve"
    requires_fit: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _active_operated_stypes(self, table: TableTensor) -> frozenset[Stype]:
        return table.active_stypes & self.operates_on_stypes

    def _active_unoperated_stypes(
        self, table: TableTensor
    ) -> frozenset[Stype]:
        return table.active_stypes - self.operates_on_stypes

    def _check_unoperated_stypes(self, table: TableTensor) -> None:
        if self.unoperated_stype_policy != "error":
            return
        for stype in self._active_unoperated_stypes(table):
            raise ValueError(
                f"{self.__class__.__name__!r} cannot preserve "
                f"{str(stype)!r} columns."
            )

    def _should_run(self, table: TableTensor) -> bool:
        if self.unoperated_stype_policy == "opaque":
            return True
        self._check_unoperated_stypes(table)
        return len(self._active_operated_stypes(table)) > 0

    @staticmethod
    def as_processor(processor: object) -> Processor:
        r"""Normalize a processor-like object to a :class:`Processor`.

        Args:
            processor: A processor-like object. A :class:`Processor` is
                returned as-is, a callable is wrapped as a stateless processor,
                and a sequence of processor-like objects is normalized to
                :class:`~sdm.processing.common.Sequential`.
        """
        from sdm.processing import Callable, Sequential  # noqa: PLC0415

        if isinstance(processor, Processor):
            return processor
        if callable(processor):
            return Callable(processor)  # type: ignore
        if isinstance(processor, Sequence) and not isinstance(processor, str):
            return Sequential(*processor)
        raise TypeError(
            f"Input must be a 'Processor', callable, or sequence of them "
            f"(got '{type(processor).__name__}')"
        )

    def _check_is_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__!r} is not fitted; "
                "call 'fit()' before."
            )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def _transform(self, table: TableTensor) -> TableTensor:
        pass

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if self.requires_fit:
            self._fit(table, generator=generator)
        return self._transform(table)

    def fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit the processor.

        Args:
            table: The table used to compute the processor state.
            generator: Pseudorandom number generator used for sampling.
        """
        if not self._should_run(table):
            return self
        if self.requires_fit:
            self._fit(table, generator=generator)
            self._fitted = True
        return self

    def transform(self, table: TableTensor) -> TableTensor:
        r"""Transform ``table``.

        Args:
            table: The table to transform.

        Returns:
            The transformed table.
        """
        if not self._should_run(table):
            return table
        self._check_is_fitted()
        return self._transform(table)

    def forward(self, table: TableTensor) -> TableTensor:
        r"""Alias of :meth:`transform`."""
        return self.transform(table)

    def fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        r"""Fit the processor and transform ``table``.

        Args:
            table: The table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed table.
        """
        if not self._should_run(table):
            return table
        out = self._fit_transform(table, generator=generator)
        if self.requires_fit:
            self._fitted = True
        return out

    def __add__(self, other: object) -> Sequential:
        from sdm.processing import Sequential  # noqa: PLC0415

        try:
            other = Processor.as_processor(other)
        except TypeError:
            return NotImplemented
        return Sequential(self, other)

    def __radd__(self, other: object) -> Sequential:
        from sdm.processing import Sequential  # noqa: PLC0415

        try:
            other = Processor.as_processor(other)
        except TypeError:
            return NotImplemented
        return Sequential(other, self)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    r"""Extend a :class:`Processor` by an inverse transformation."""

    @abc.abstractmethod
    def _inverse_transform(self, table: TableTensor) -> TableTensor: ...

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        r"""Apply the inverse transformation to ``table``.

        Args:
            table: The table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~Processor.transform`.
        """
        self._check_is_fitted()
        return self._inverse_transform(table)

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...
