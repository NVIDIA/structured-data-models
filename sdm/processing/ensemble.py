from __future__ import annotations

import copy
from typing import cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import EnsembleTable, TableTensor


class VariableSchemaBatchMixin:
    r"""Process a leading batch whose members may produce different schemas."""

    def fit_transform_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, ...]:
        r"""Fit and transform each leading table position."""
        self.fit_batch(table, generator=generator)
        return self.transform_batch(table)

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit state for each leading table position."""
        raise NotImplementedError

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        r"""Transform each leading table position."""
        raise NotImplementedError


class EnsembleProcessor(Processor):
    r"""A Processor whose semantics depend on logical ensemble members."""

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.fit_transform_ensemble(
            EnsembleTable(table, num_members=1),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        ensemble = EnsembleTable(table, num_members=1)
        if not self.requires_fit:
            return self.fit_transform_ensemble(ensemble).representation(0)
        return self.transform_ensemble(ensemble).representation(0)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if (
            type(self)._fit is not EnsembleProcessor._fit
            or type(self)._transform is not EnsembleProcessor._transform
        ):
            return super()._fit_transform(table, generator=generator)
        return self.fit_transform_ensemble(
            EnsembleTable(table, num_members=1),
            generator=generator,
        ).representation(0)

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        r"""Fit and transform an ensemble table."""
        raise NotImplementedError

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Transform an ensemble table with fitted state."""
        raise NotImplementedError

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Inverse-transform an ensemble table."""
        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute "
            "'inverse_transform_ensemble'"
        )


class EnsembleProcessorAdapter(EnsembleProcessor):
    r"""Adapt an ordinary Processor to packed ensemble representations.

    One fitted Processor copy owns each packed input representation. Processors
    implementing :class:`VariableSchemaBatchMixin` may split that
    representation into multiple schemas.
    """

    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.template = processor
        self.requires_fit = processor.requires_fit
        self.processors = torch.nn.ModuleList()
        self._fitted_member_locations: tuple[tuple[int, int], ...] = ()
        self._fitted_packed_sizes: tuple[int, ...] = ()

    @classmethod
    def adapt(cls, processor: Processor) -> EnsembleProcessor:
        r"""Return `processor` or an adapter for an ordinary Processor."""
        if isinstance(processor, EnsembleProcessor):
            return processor
        return cls(processor)

    def _check_rows(
        self,
        before: TableTensor,
        after: TableTensor,
    ) -> None:
        if before.size(-2) != after.size(-2):
            raise ValueError(
                f"{self.template.__class__.__name__!r} changes the row "
                "dimension and is not supported in ensemble Recipes."
            )

    def fit_transform_ensemble(  # noqa: D102
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.processors = torch.nn.ModuleList()
        self._fitted_member_locations = table._member_locations
        self._fitted_packed_sizes = tuple(
            group.size(0) for group in table.iter_packed_representations()
        )

        if not isinstance(self.template, VariableSchemaBatchMixin):
            outputs = []
            for group in table.iter_packed_representations():
                processor = copy.deepcopy(self.template)
                output = processor.fit_transform(
                    group,
                    generator=generator,
                )
                self._check_rows(group, output)
                self.processors.append(processor)
                outputs.append(output)
            return table._replace_packed_representations(tuple(outputs))

        representations: list[TableTensor] = []
        offsets: list[int] = []
        for group in table.iter_packed_representations():
            processor = copy.deepcopy(self.template)
            offsets.append(len(representations))
            outputs = cast(
                VariableSchemaBatchMixin,
                processor,
            ).fit_transform_batch(
                group,
                generator=generator,
            )
            if len(outputs) != group.size(0):
                raise RuntimeError(
                    f"{processor.__class__.__name__!r} must return one "
                    "table per leading input position."
                )
            for before, output in zip(group, outputs, strict=True):
                self._check_rows(before, output)
            self.processors.append(processor)
            representations.extend(outputs)

        return EnsembleTable.pack(
            representations,
            tuple(
                offsets[group] + variant
                for group, variant in table._member_locations
            ),
            member_ids=table._member_ids,
        )

    def transform_ensemble(  # noqa: D102
        self, table: EnsembleTable
    ) -> EnsembleTable:
        if not isinstance(self.template, VariableSchemaBatchMixin):
            outputs = []
            for group, processor in zip(
                table.iter_packed_representations(),
                self.processors,
                strict=True,
            ):
                output = cast(Processor, processor).transform(group)
                self._check_rows(group, output)
                outputs.append(output)
            return table._replace_packed_representations(tuple(outputs))

        representations: list[TableTensor] = []
        offsets: list[int] = []
        for group, processor in zip(
            table.iter_packed_representations(),
            self.processors,
            strict=True,
        ):
            offsets.append(len(representations))
            outputs = cast(
                VariableSchemaBatchMixin,
                processor,
            ).transform_batch(group)
            if len(outputs) != group.size(0):
                raise RuntimeError(
                    f"{processor.__class__.__name__!r} must return one "
                    "table per leading input position."
                )
            for before, output in zip(group, outputs, strict=True):
                self._check_rows(before, output)
            representations.extend(outputs)

        return EnsembleTable.pack(
            representations,
            tuple(
                offsets[group] + variant
                for group, variant in table._member_locations
            ),
            member_ids=table._member_ids,
        )

    def inverse_transform_ensemble(  # noqa: D102
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        if isinstance(self.template, VariableSchemaBatchMixin):
            raise TypeError("Variable-schema processors are not invertible.")

        if self._fitted_packed_sizes == (1,):
            processor = self.processors[0]
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            outputs = tuple(
                processor.inverse_transform(group)
                for group in table.iter_packed_representations()
            )
            return table._replace_packed_representations(outputs)

        members_by_representation = [
            [[] for _ in range(size)] for size in self._fitted_packed_sizes
        ]
        for member, (group, variant) in enumerate(
            self._fitted_member_locations
        ):
            members_by_representation[group][variant].append(member)

        representations: list[TableTensor] = []
        member_representation_ids = [0] * table.num_members
        for processor, group_members in zip(
            self.processors,
            members_by_representation,
            strict=True,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            rounds = max(len(members) for members in group_members)
            round_tables = []
            for round_index in range(rounds):
                variants = tuple(
                    table.representation(
                        members[min(round_index, len(members) - 1)]
                    )
                    for members in group_members
                )
                round_tables.append(
                    EnsembleTable.pack(
                        variants,
                        tuple(range(len(variants))),
                    ).materialize()
                )
            batch = EnsembleTable.pack(
                round_tables,
                tuple(range(len(round_tables))),
            ).materialize()
            restored = processor.inverse_transform(batch)
            self._check_rows(batch, restored)
            for variant, members in enumerate(group_members):
                for round_index, member in enumerate(members):
                    member_representation_ids[member] = len(representations)
                    representations.append(restored[round_index, variant])

        return EnsembleTable.pack(
            representations,
            member_representation_ids,
            member_ids=table._member_ids,
        )
