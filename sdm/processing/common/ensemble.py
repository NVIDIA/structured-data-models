import copy
from itertools import repeat
from typing import cast

import torch
from torch.nn import ModuleList

from sdm import EnsembleTable, Stype
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    InvertibleMixin,
    Processor,
)


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`~sdm.processing.ensemble.EnsembleProcessor`. It copies and fits the
    processor separately for each group of compatible tables in an
    :class:`~sdm.EnsembleTable`.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement
    :class:`~sdm.processing.ensemble.EnsembleProcessor` directly. Inverse
    transformation requires the wrapped processor to implement
    :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to fit separately for each ensemble table group.
    """

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._processors: ModuleList[Processor] = ModuleList()
        self._processor_locations: tuple[tuple[int, int], ...] = ()

    def get_extra_state(
        self,
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        r""":meta private:"""  # noqa: D415
        return len(self._processors), self._processor_locations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        num_processors, self._processor_locations = cast(
            tuple[int, tuple[tuple[int, int], ...]], state
        )
        self._processors = ModuleList(
            [copy.deepcopy(self.processor) for _ in range(num_processors)]
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
        self._processor_locations = ensemble_table._locations
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            processor.fit(group, generator=generator)
            self._processors.append(processor)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        self._processor_locations = ensemble_table._locations
        outputs = []
        self._processors = ModuleList()
        for group_id, group in enumerate(ensemble_table):
            if group_id == 0:
                processor = self.processor
            else:
                processor = copy.deepcopy(self.processor)
            outputs.append(processor.fit_transform(group, generator=generator))
            self._processors.append(processor)
        return ensemble_table.replace_groups(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            ensemble_table = ensemble_table._split_groups(
                self._processor_locations
            )
            processors = self._processors_for_groups(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = [
            processor.transform(group)
            for group, processor in zip(
                ensemble_table, processors, strict=True
            )
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            ensemble_table = ensemble_table._split_groups(
                self._processor_locations
            )
            processors = self._processors_for_groups(ensemble_table)
        else:
            processors = repeat(self.processor, ensemble_table.num_groups)

        outputs = []
        for group, processor in zip(ensemble_table, processors, strict=True):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def _processors_for_groups(
        self,
        ensemble_table: EnsembleTable,
    ) -> tuple[Processor, ...]:
        if ensemble_table._locations == self._processor_locations:
            return tuple(self._processors)

        processor_id_by_group: dict[int, int] = {}
        member_ids_by_processor: dict[int, list[int]] = {}
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            processor_id, _ = self._processor_locations[member_id]
            processor_id_by_group[group_id] = processor_id
            member_ids_by_processor.setdefault(processor_id, []).append(
                member_id
            )

        for member_ids in member_ids_by_processor.values():
            fitted_positions = tuple(
                self._processor_locations[member_id][1]
                for member_id in member_ids
            )
            if len(set(fitted_positions)) == 1:
                continue

            current_locations = tuple(
                ensemble_table._locations[member_id]
                for member_id in member_ids
            )
            current_group_ids = {group_id for group_id, _ in current_locations}
            current_positions = tuple(
                position for _, position in current_locations
            )
            if (
                len(current_group_ids) > 1
                or current_positions != fitted_positions
            ):
                raise RuntimeError(
                    "Cannot apply position-dependent fitted processor state "
                    "after its ensemble group was split or reordered"
                )

        return tuple(
            self._processors[processor_id_by_group[group_id]]
            for group_id in range(ensemble_table.num_groups)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
