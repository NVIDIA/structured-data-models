from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, TypeAlias, cast

import torch
from typing_extensions import Self

from sdm import Stype, TableTensor

if TYPE_CHECKING:
    from sdm.processing import Sequential

SupportedStypes: TypeAlias = frozenset[Stype]


class Processor(torch.nn.Module, abc.ABC):
    r"""Base processor for tensor-aware table transformations.

    A :class:`Processor` defines a reusable transformation on
    :class:`~sdm.tensor.TableTensor` for feature, target and output
    preprocessing.
    A :class:`Processor` learns any required state via :meth:`fit`, and applies
    the transformation via :meth:`transform`.
    """

    supported_stypes: ClassVar[SupportedStypes]
    requires_fit: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False

    def _check_supported_stypes(self, table: TableTensor) -> None:
        supported_stypes = self.supported_stypes
        for stype, columns in table.columns.items():
            if stype not in supported_stypes and len(columns) > 0:
                # TODO: Include all invalid columns in the error message
                raise ValueError(
                    f"{self.__class__.__name__!r} does not support "
                    f"{stype.value!r} columns."
                )

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
        self._check_supported_stypes(table)
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
        self._check_supported_stypes(table)
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
        self._check_supported_stypes(table)
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


class VariableSchemaProcessor(Processor):
    r"""Process each entry of a batched table independently.

    Each input batch entry may result in a different column schema. Batch
    transform methods therefore return one :class:`TableTensor` per entry.
    """

    # TODO: Add lifecycle coverage with the first concrete implementation.

    def __init__(self) -> None:
        super().__init__()
        self._batch_fitted = False

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        batch = cast(TableTensor, table.unsqueeze(0))
        self._fit_batch(batch, generator=generator)

    def _transform(self, table: TableTensor) -> TableTensor:
        batch = cast(TableTensor, table.unsqueeze(0))
        return self._transform_batch(batch)[0]

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        batch = cast(TableTensor, table.unsqueeze(0))
        return self._fit_transform_batch(batch, generator=generator)[0]

    def _check_is_batch_fitted(self) -> None:
        if self.requires_fit and not self._batch_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__!r} is not fitted for batch "
                "transforms; call 'fit_batch()' before."
            )

    def _fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def _transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]: ...

    def _fit_transform_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, ...]:
        if self.requires_fit:
            self._fit_batch(table, generator=generator)
        return self._transform_batch(table)

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit the processor on a batched table.

        Args:
            table: Table with shape ``[B, ..., R, C]``, where ``B`` contains
                independently processed batch entries, ``R`` is the number of
                rows, and ``C`` is the number of columns.
            generator: Pseudorandom number generator used for sampling.
        """
        self._check_supported_stypes(table)
        if self.requires_fit:
            self._fit_batch(table, generator=generator)
            self._batch_fitted = True
        return self

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        r"""Transform every entry of a batched table independently.

        Args:
            table: Table with shape ``[B, ..., R, C]``, where ``B`` contains
                independently processed batch entries, ``R`` is the number of
                rows, and ``C`` is the number of columns.

        Returns:
            One transformed table per input batch entry.
        """
        self._check_supported_stypes(table)
        self._check_is_batch_fitted()
        return self._transform_batch(table)

    def fit_transform_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, ...]:
        r"""Fit and independently transform every entry of a batched table.

        Args:
            table: Table with shape ``[B, ..., R, C]``, where ``B`` contains
                independently processed batch entries, ``R`` is the number of
                rows, and ``C`` is the number of columns.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            One transformed table per input batch entry.
        """
        self._check_supported_stypes(table)
        out = self._fit_transform_batch(table, generator=generator)
        if self.requires_fit:
            self._batch_fitted = True
        return out
