from __future__ import annotations

import abc

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import EnsembleTable, TableTensor


class EnsembleProcessor(Processor):
    """Base processor for transformations defined over complete ensembles.

    Implementations receive a complete :class:`~sdm.tensor.EnsembleTable` and
    may change its member count or member-to-representation mapping.
    The inherited :class:`~sdm.tensor.TableTensor` lifecycle treats its input
    as an ensemble with one member and therefore requires one output member.
    """

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_transform_ensemble(
            EnsembleTable(table, num_members=1),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        output = self._transform_ensemble(EnsembleTable(table, num_members=1))
        return self._single_member(output)

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
        return self._single_member(output)

    @abc.abstractmethod
    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable: ...

    @abc.abstractmethod
    def _transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable: ...

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        """Fit the processor and transform an ensemble table.

        Args:
            table: Ensemble table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed ensemble table.
        """
        for packed in table.iter_packed_representations():
            self._check_supported_stypes(packed)
        output = self._fit_transform_ensemble(
            table,
            generator=generator,
        )
        if self.requires_fit:
            self._fitted = True
        return output

    def transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Transform an ensemble table with fitted state.

        Args:
            table: Ensemble table to transform.

        Returns:
            The transformed ensemble table.
        """
        for packed in table.iter_packed_representations():
            self._check_supported_stypes(packed)
        self._check_is_fitted()
        return self._transform_ensemble(table)

    @staticmethod
    def _single_member(table: EnsembleTable) -> TableTensor:
        if table.num_members != 1:
            raise RuntimeError(
                "An EnsembleProcessor used with a TableTensor must return "
                "exactly one member."
            )
        return table.representation(0)


class EnsembleInvertibleMixin(InvertibleMixin):
    r"""Extend a :class:`EnsembleProcessor` by an inverse transformation."""

    @abc.abstractmethod
    def _inverse_transform_ensemble(
        self, table: EnsembleTable
    ) -> EnsembleTable: ...

    def inverse_transform_ensemble(
        self, table: EnsembleTable
    ) -> EnsembleTable:
        r"""Apply the inverse transformation to ``ensemble table``.

        Args:
            table: The ensemble table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~EnsembleProcessor.transform_ensemble`.
        """
        self._check_is_fitted()
        return self._inverse_transform_ensemble(table)
