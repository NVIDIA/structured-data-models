from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    EnsembleTable,
)
from sdm.tensor import TableTensor


class ShuffleColumns(EnsembleProcessor, InvertibleMixin):
    """Permute the numerical feature columns.

    The permutation is drawn when the processor is fitted; pass
    ``generator`` to ``fit()`` to make it reproducible. Convert
    non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            columns by a drawn offset, ``"random"`` draws a permutation, and
            ``"latin"`` uses an ensemble planner (or identity for scalar
            execution).
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: Literal["shift", "random", "latin"] = "shift",
    ) -> None:
        super().__init__()
        if method not in {"shift", "random", "latin"}:
            raise ValueError("method must be 'shift', 'random', or 'latin'")
        self.method = method
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )
        self.processors = torch.nn.ModuleList()
        self._member_to_processor: tuple[int, ...] = ()
        self._processor_positions: tuple[int, ...] = ()
        self._indices: tuple[int, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        n_features = table.numerical.size(-1)
        device = table.numerical.device
        random_device = device if generator is None else generator.device
        if n_features <= 1 or self.method == "latin":
            permutation = torch.arange(n_features, device=random_device)
        elif self.method == "shift":
            offset = torch.randint(
                n_features,
                (1,),
                generator=generator,
                device=random_device,
            )
            permutation = (
                torch.arange(n_features, device=random_device) + offset
            ) % n_features
        else:
            permutation = torch.randperm(
                n_features,
                generator=generator,
                device=random_device,
            )
        self._indices = tuple(permutation.tolist())
        self.permutation = permutation.to(device)

    @staticmethod
    def _set_permutation(
        processor: ShuffleColumns,
        table: TableTensor,
        permutation: Sequence[int],
    ) -> None:
        permutation = tuple(permutation)
        if sorted(permutation) != list(range(table.numerical.size(-1))):
            raise ValueError(
                "The ensemble plan returned an invalid column permutation."
            )
        processor.permutation = torch.tensor(
            permutation,
            dtype=torch.long,
            device=table.device,
        )
        processor._indices = permutation
        processor._fitted = True

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit member permutations and reuse proven-equal results."""
        planned = (
            context._plan.column_permutations(
                member_ids=context.member_ids,
                num_columns=tuple(
                    table[position].numerical.size(-1)
                    for position in range(table.num_members)
                ),
                table_scope=context.table_scope,
            )
            if context._plan is not None
            else None
        )
        if planned is not None and len(planned) != table.num_members:
            raise ValueError(
                "The ensemble plan must return one permutation per member."
            )

        self.processors = torch.nn.ModuleList()
        variants: list[TableTensor] = []
        positions: list[int] = []
        member_to_processor: list[int] = []
        keys: dict[tuple[object, ...], int] = {}
        for position, member_id in enumerate(context.member_ids):
            before = table[position]
            processor = self.__class__(method=self.method)
            if planned is None:
                processor.fit(
                    before,
                    generator=context.generator_for(member_id),
                )
            else:
                self._set_permutation(
                    processor,
                    before,
                    planned[position],
                )

            key = (
                table.member_to_variant[position],
                processor._indices,
            )
            processor_index = keys.get(key)
            if processor_index is None:
                processor_index = len(variants)
                keys[key] = processor_index
                self.processors.append(processor)
                positions.append(position)
                variants.append(processor.transform(before))
            member_to_processor.append(processor_index)

        self._member_to_processor = tuple(member_to_processor)
        self._processor_positions = tuple(positions)
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=self._member_to_processor,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Apply the fitted member permutations."""
        variants = tuple(
            cast(ShuffleColumns, processor).transform(table[position])
            for processor, position in zip(
                self.processors,
                self._processor_positions,
            )
        )
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=self._member_to_processor,
        )

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Invert each output with its fitted member permutation."""
        variants = tuple(
            cast(
                ShuffleColumns,
                self.processors[processor_index],
            ).inverse_transform(table[position])
            for position, processor_index in enumerate(
                self._member_to_processor
            )
        )
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=tuple(range(table.num_members)),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Reorder the numerical block with the fitted permutation."""
        return self._permute(table, self.permutation, self._indices)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        inverse = tuple(
            sorted(range(len(self._indices)), key=self._indices.__getitem__)
        )
        return self._permute(table, self.permutation.argsort(), inverse)

    def _permute(
        self,
        table: TableTensor,
        permutation: Tensor,
        indices: tuple[int, ...],
    ) -> TableTensor:
        return table.__class__(
            columns={
                Stype.numerical.value: tuple(
                    table.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=table.numerical.index_select(-1, permutation),
        )
