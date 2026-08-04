from __future__ import annotations

import abc
import copy
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
    """Apply an ordinary processor to compatible table groups.

    The adapter owns one fitted processor copy per group and preserves the
    member-to-table mapping. It only supports processors whose output remains
    grouped with the same leading size.

    Args:
        processor: Ordinary processor to adapt.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.template = processor
        self.requires_fit = processor.requires_fit
        self.processors = torch.nn.ModuleList()

    @classmethod
    def adapt(cls, processor: Processor) -> EnsembleProcessor:
        """Return an ensemble processor for ``processor``.

        Args:
            processor: Processor to normalize.
        """
        if isinstance(processor, EnsembleProcessor):
            return processor
        return cls(processor)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.processors = torch.nn.ModuleList()
        for group in ensemble_table:
            processor = copy.deepcopy(self.template)
            processor.fit(group, generator=generator)
            self.processors.append(processor)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.processors = torch.nn.ModuleList()
        outputs = []
        for group in ensemble_table:
            processor = copy.deepcopy(self.template)
            output = processor.fit_transform(group, generator=generator)
            self._check_output(group, output)
            self.processors.append(processor)
            outputs.append(output)
        return ensemble_table._replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not self.requires_fit and len(self.processors) == 0:
            self.processors = torch.nn.ModuleList(
                copy.deepcopy(self.template) for _ in ensemble_table
            )

        outputs = []
        for group, processor in zip(
            ensemble_table,
            self.processors,
            strict=True,
        ):
            output = cast(Processor, processor).transform(group)
            self._check_output(group, output)
            outputs.append(output)
        return ensemble_table._replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        outputs = []
        for group, processor in zip(
            ensemble_table,
            self.processors,
            strict=True,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            output = processor.inverse_transform(group)
            self._check_output(group, output)
            outputs.append(output)
        return ensemble_table._replace_groups(outputs)

    @staticmethod
    def _check_output(before: TableTensor, after: TableTensor) -> None:
        if before.size(-2) != after.size(-2):
            raise ValueError(
                "An adapted Processor must preserve the row dimension."
            )

    def __repr__(self, *, indent: int = 0) -> str:
        return self.template.__repr__(indent=indent)
