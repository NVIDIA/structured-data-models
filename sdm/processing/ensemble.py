from __future__ import annotations

import abc
import copy
from collections.abc import Iterable
from itertools import repeat
from typing import cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


class EnsembleProcessor(Processor):
    r"""Base processor for ensemble-aware table transformations.

    An :class:`EnsembleProcessor` defines a reusable transformation on
    :class:`~sdm.tensor.EnsembleTable` for feature, target and output
    pre/post-processing across ensemble members.
    An :class:`EnsembleProcessor` learns any required state via
    :meth:`fit_ensemble` and applies the transformation via
    :meth:`transform_ensemble`. :meth:`fit_ensemble`,
    :meth:`transform_ensemble`, and :meth:`fit_transform_ensemble`
    are no-ops for supported stypes with empty blocks.

    As a :class:`~sdm.processing.base.Processor`, it also accepts a
    :class:`~sdm.tensor.TableTensor` and processes it as an ensemble
    with one member.
    """

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_ensemble(
            EnsembleTable(table, num_members=1),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        output = self._transform_ensemble(EnsembleTable(table, num_members=1))
        return output.table(0)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        output = self._fit_transform_ensemble(
            EnsembleTable(table, num_members=1),
            generator=generator,
        )
        return output.table(0)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable: ...

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self.requires_fit:
            self._fit_ensemble(ensemble_table, generator=generator)
        return self._transform_ensemble(ensemble_table)

    def fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        """Fit the processor on an ensemble table.

        Args:
            ensemble_table: Ensemble table used to compute the processor state.
            generator: Pseudorandom number generator used for sampling.
        """
        for group in ensemble_table:
            self._check_supported_stypes(group)
        if not any(
            group.active_stypes & self.supported_stypes
            for group in ensemble_table
        ):
            return self
        if self.requires_fit:
            self._fit_ensemble(ensemble_table, generator=generator)
            self._fitted = True
        return self

    def transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        """Transform an ensemble table with fitted state.

        Args:
            ensemble_table: Ensemble table to transform.

        Returns:
            The transformed ensemble table.
        """
        for group in ensemble_table:
            self._check_supported_stypes(group)
        if not any(
            group.active_stypes & self.supported_stypes
            for group in ensemble_table
        ):
            return ensemble_table
        self._check_is_fitted()
        return self._transform_ensemble(ensemble_table)

    def fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        """Fit the processor and transform an ensemble table.

        Args:
            ensemble_table: Ensemble table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed ensemble table.
        """
        for group in ensemble_table:
            self._check_supported_stypes(group)
        if not any(
            group.active_stypes & self.supported_stypes
            for group in ensemble_table
        ):
            return ensemble_table
        output = self._fit_transform_ensemble(
            ensemble_table,
            generator=generator,
        )
        if self.requires_fit:
            self._fitted = True
        return output


class EnsembleInvertibleMixin(InvertibleMixin):
    r"""Extend a :class:`EnsembleProcessor` by an inverse transformation."""

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        output = self._inverse_transform_ensemble(
            EnsembleTable(table, num_members=1)
        )
        return output.table(0)

    @abc.abstractmethod
    def _inverse_transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable: ...

    def inverse_transform_ensemble(
        self, ensemble_table: EnsembleTable
    ) -> EnsembleTable:
        r"""Apply the inverse transformation to ``ensemble_table``.

        Args:
            ensemble_table: The ensemble table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~EnsembleProcessor.transform_ensemble`.
        """
        self._check_is_fitted()
        return self._inverse_transform_ensemble(ensemble_table)


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`EnsembleProcessor`. It copies and fits the processor separately
    for each group of compatible tables in an
    :class:`~sdm.tensor.EnsembleTable`.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement :class:`EnsembleProcessor`
    directly. Inverse transformation requires the wrapped processor to
    implement :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to fit separately for each ensemble table group.
    """

    supported_stypes = frozenset[Stype](Stype)

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit
        self._group_processors = torch.nn.ModuleList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        processors = []
        for group in ensemble_table:
            processor = copy.deepcopy(self.processor)
            processor.fit(group, generator=generator)
            processors.append(processor)
        self._group_processors = torch.nn.ModuleList(processors)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        processors = []
        outputs = []
        for group in ensemble_table:
            processor = copy.deepcopy(self.processor)
            outputs.append(processor.fit_transform(group, generator=generator))
            processors.append(processor)
        self._group_processors = torch.nn.ModuleList(processors)
        return ensemble_table.replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        processors = (
            cast(Iterable[Processor], self._group_processors)
            if self.requires_fit
            else repeat(self.processor, ensemble_table.num_groups)
        )
        outputs = [
            processor.transform(group)
            for group, processor in zip(
                ensemble_table,
                processors,
                strict=True,
            )
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        processors = (
            cast(Iterable[Processor], self._group_processors)
            if self.requires_fit
            else repeat(self.processor, ensemble_table.num_groups)
        )
        outputs = []
        for group, processor in zip(
            ensemble_table,
            processors,
            strict=True,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
