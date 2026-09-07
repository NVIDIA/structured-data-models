import copy
from collections.abc import Iterable
from typing import cast

import torch
from torch.nn import ModuleList

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`~sdm.processing.ensemble.EnsembleProcessor`. Fitted processors are
    copied and fitted separately for each logical member in an
    :class:`~sdm.EnsembleTable`.

    Stateless processors transform each physically stored table once, so
    members that share an input also share its transformed representation.
    A subsequent fitted processor fans that representation out through its
    member-specific states.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement
    :class:`~sdm.processing.ensemble.EnsembleProcessor` directly. Inverse
    transformation requires the wrapped processor to implement
    :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to adapt. Fitted processors learn separate state
            for each ensemble member.
    """

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._member_processors: ModuleList[Processor] = ModuleList()

    def get_extra_state(self) -> int:
        r""":meta private:"""  # noqa: D415
        return len(self._member_processors)

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        num_members = cast(int, state)
        self._member_processors = ModuleList(
            [copy.deepcopy(self.processor) for _ in range(num_members)]
        )

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return self.processor.handles_stypes

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        processors: ModuleList[Processor] = ModuleList()
        for member_id in range(ensemble_table.num_members):
            if member_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            processor.fit(
                ensemble_table.table(member_id),
                generator=generator,
            )
            processors.append(processor)
        self._member_processors = processors

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        outputs: list[TableTensor] = []
        processors: ModuleList[Processor] = ModuleList()
        for member_id in range(ensemble_table.num_members):
            if member_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            outputs.append(
                processor.fit_transform(
                    ensemble_table.table(member_id),
                    generator=generator,
                )
            )
            processors.append(processor)
        self._member_processors = processors
        return EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=range(len(outputs)),
        )

    def _processors_for(
        self,
        ensemble_table: EnsembleTable,
    ) -> Iterable[Processor]:
        if len(self._member_processors) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._member_processors)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )
        return self._member_processors

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return ensemble_table.replace_groups(
                [self.processor.transform(group) for group in ensemble_table]
            )

        outputs = [
            processor.transform(ensemble_table.table(member_id))
            for member_id, processor in enumerate(
                self._processors_for(ensemble_table)
            )
        ]
        return EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=range(len(outputs)),
        )

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not self.requires_fit:
            if not isinstance(self.processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            return ensemble_table.replace_groups(
                [
                    self.processor.inverse_transform(group)
                    for group in ensemble_table
                ]
            )

        outputs: list[TableTensor] = []
        for member_id, processor in enumerate(
            self._processors_for(ensemble_table)
        ):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(
                processor.inverse_transform(ensemble_table.table(member_id))
            )
        return EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=range(len(outputs)),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
