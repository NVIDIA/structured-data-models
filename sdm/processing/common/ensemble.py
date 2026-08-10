import copy
from itertools import repeat

import torch
from torch.nn import ModuleList

from sdm import Stype
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)
from sdm.processing.base import UnoperatedStypePolicy
from sdm.tensor import EnsembleTable


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

    @property
    def operates_on_stypes(self) -> frozenset[Stype]:
        """Semantic types operated on by the wrapped processor."""
        return self.processor.operates_on_stypes

    @property
    def unoperated_stype_policy(self) -> UnoperatedStypePolicy:
        """Policy delegated to the wrapped processor."""
        return self.processor.unoperated_stype_policy

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._group_processors: ModuleList[Processor] = ModuleList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._group_processors = ModuleList()
        for i, group in enumerate(ensemble_table):
            if i == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            processor.fit(group, generator=generator)
            self._group_processors.append(processor)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        outputs = []
        self._group_processors = ModuleList()
        for i, group in enumerate(ensemble_table):
            if i == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            outputs.append(processor.fit_transform(group, generator=generator))
            self._group_processors.append(processor)
        return ensemble_table.replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._group_processors
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = [
            processor.transform(group)
            for group, processor in zip(ensemble_table, processors)
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            processors = self._group_processors
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = []
        for group, processor in zip(ensemble_table, processors):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
