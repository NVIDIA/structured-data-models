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
from sdm.tensor import EnsembleTable


class EnsembleProcessorAdapter(EnsembleProcessor, EnsembleInvertibleMixin):
    """Adapt an ordinary processor to ensemble-aware processing.

    The adapter turns a :class:`~sdm.processing.base.Processor` into an
    :class:`EnsembleProcessor`. It copies and fits the processor separately
    for each unique stored table in an
    :class:`~sdm.tensor.EnsembleTable`.

    The wrapped processor must preserve row and leading dimensions as required
    by the :class:`~sdm.processing.base.Processor` contract. A processor that
    changes the ensemble structure must implement :class:`EnsembleProcessor`
    directly. Inverse transformation requires the wrapped processor to
    implement :class:`~sdm.processing.base.InvertibleMixin`.

    Args:
        processor: Processor to fit separately for each stored table.
    """

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.processor = processor
        self.requires_fit = processor.requires_fit

        self._table_processors: ModuleList[Processor] = ModuleList()
        self._member_processor_ids: tuple[int, ...] = ()

    @staticmethod
    def _stored_tables(
        ensemble_table: EnsembleTable,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Map logical members to unique stored tables."""
        processor_id_by_location: dict[tuple[int, int], int] = {}
        member_processor_ids = []
        representative_member_ids = []

        # A location is the canonical identity of a stored table within an
        # EnsembleTable. Assign our own IDs in logical member order so fitted
        # state never depends on physical group order.
        for member_id, location in enumerate(ensemble_table._locations):
            processor_id = processor_id_by_location.get(location)
            if processor_id is None:
                processor_id = len(representative_member_ids)
                processor_id_by_location[location] = processor_id
                representative_member_ids.append(member_id)
            member_processor_ids.append(processor_id)

        return tuple(member_processor_ids), tuple(representative_member_ids)

    def _initialize_processors(self, num_tables: int) -> None:
        self._table_processors = ModuleList(
            copy.deepcopy(self.processor) for _ in range(num_tables)
        )

    def _check_num_members(self, ensemble_table: EnsembleTable) -> None:
        if len(self._member_processor_ids) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._member_processor_ids)} ensemble members, but "
                f"got {ensemble_table.num_members}."
            )

    def _transform_with_fitted_processors(
        self,
        ensemble_table: EnsembleTable,
        *,
        inverse: bool = False,
    ) -> EnsembleTable:
        self._check_num_members(ensemble_table)
        if ensemble_table.num_members == 0:
            return ensemble_table

        outputs = []
        member_output_ids = []
        output_id_by_state: dict[tuple[tuple[int, int], int], int] = {}

        for member_id, processor_id in enumerate(self._member_processor_ids):
            location = ensemble_table._locations[member_id]
            state = (location, processor_id)
            output_id = output_id_by_state.get(state)
            if output_id is None:
                processor = self._table_processors[processor_id]
                table = ensemble_table.table(member_id)
                if inverse:
                    if not isinstance(processor, InvertibleMixin):
                        raise AttributeError(
                            f"{self.processor.__class__.__name__!r} object "
                            "has no attribute 'inverse_transform'"
                        )
                    output = processor.inverse_transform(table)
                else:
                    output = processor.transform(table)
                output_id = len(outputs)
                output_id_by_state[state] = output_id
                outputs.append(output)
            member_output_ids.append(output_id)

        return EnsembleTable.from_tables(outputs, member_output_ids)

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
        (
            self._member_processor_ids,
            representative_member_ids,
        ) = self._stored_tables(ensemble_table)
        self._initialize_processors(len(representative_member_ids))
        for processor, member_id in zip(
            self._table_processors,
            representative_member_ids,
            strict=True,
        ):
            processor.fit(
                ensemble_table.table(member_id),
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if not self.requires_fit:
            return self._transform_ensemble(ensemble_table)

        (
            self._member_processor_ids,
            representative_member_ids,
        ) = self._stored_tables(ensemble_table)
        self._initialize_processors(len(representative_member_ids))
        outputs = [
            processor.fit_transform(
                ensemble_table.table(member_id),
                generator=generator,
            )
            for processor, member_id in zip(
                self._table_processors,
                representative_member_ids,
                strict=True,
            )
        ]
        if ensemble_table.num_members == 0:
            return ensemble_table
        return EnsembleTable.from_tables(
            outputs,
            self._member_processor_ids,
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            return self._transform_with_fitted_processors(ensemble_table)

        outputs = [
            processor.transform(group)
            for group, processor in zip(
                ensemble_table,
                repeat(self.processor, ensemble_table.num_groups),
                strict=True,
            )
        ]
        return ensemble_table.replace_groups(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.requires_fit:
            return self._transform_with_fitted_processors(
                ensemble_table,
                inverse=True,
            )

        outputs = []
        for group, processor in zip(
            ensemble_table,
            repeat(self.processor, ensemble_table.num_groups),
            strict=True,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise AttributeError(
                    f"{self.processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform'"
                )
            outputs.append(processor.inverse_transform(group))
        return ensemble_table.replace_groups(outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)
