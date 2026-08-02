from __future__ import annotations

import abc
import copy
from typing import cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
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


class EnsembleProcessorAdapter(EnsembleProcessor):
    """Apply an ordinary processor to packed ensemble representations.

    The adapter owns one fitted processor copy per packed representation and
    preserves the member-to-representation mapping. It only supports
    processors whose output remains packed with the same leading size.

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

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.processors = torch.nn.ModuleList()
        outputs = []
        for packed in table.iter_packed_representations():
            processor = copy.deepcopy(self.template)
            output = processor.fit_transform(packed, generator=generator)
            self._check_output(packed, output)
            self.processors.append(processor)
            outputs.append(output)
        return table._replace_packed_representations(outputs)

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        processors = self.processors
        if not self.requires_fit and len(processors) == 0:
            processors = torch.nn.ModuleList(
                copy.deepcopy(self.template)
                for _ in table.iter_packed_representations()
            )

        outputs = []
        for packed, processor in zip(
            table.iter_packed_representations(),
            processors,
            strict=True,
        ):
            output = cast(Processor, processor).transform(packed)
            self._check_output(packed, output)
            outputs.append(output)
        return table._replace_packed_representations(outputs)

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Apply the fitted inverse to packed representations.

        Args:
            table: Ensemble table in the transformed representation.

        Returns:
            Ensemble table restored to its representation before transform.
        """
        outputs = []
        for packed, processor in zip(
            table.iter_packed_representations(),
            self.processors,
            strict=True,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            output = processor.inverse_transform(packed)
            self._check_output(packed, output)
            outputs.append(output)
        return table._replace_packed_representations(outputs)

    @staticmethod
    def _check_output(before: TableTensor, after: TableTensor) -> None:
        if before.size(-2) != after.size(-2):
            raise ValueError(
                "An adapted Processor must preserve the row dimension."
            )

    def __repr__(self, *, indent: int = 0) -> str:
        return self.template.__repr__(indent=indent)
