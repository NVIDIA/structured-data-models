from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin
from sdm.processing.ensemble import EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor


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
        *,
        _ensemble_permutations: Callable[..., Sequence[Sequence[int]]]
        | None = None,
    ) -> None:
        super().__init__()
        if method not in {"shift", "random", "latin"}:
            raise ValueError("method must be 'shift', 'random', or 'latin'")
        self.method = method
        self._ensemble_permutations = _ensemble_permutations
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

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        r"""Fit member permutations and reuse proven-equal results."""
        planned = (
            self._ensemble_permutations(
                table._member_ids,
                tuple(
                    table.representation(position).numerical.size(-1)
                    for position in range(table.num_members)
                ),
            )
            if self._ensemble_permutations is not None
            else None
        )
        self.processors = torch.nn.ModuleList()
        representations: list[TableTensor] = []
        positions: list[int] = []
        member_to_processor: list[int] = []
        keys: dict[tuple[object, ...], int] = {}
        for position, _ in enumerate(table._member_ids):
            before = table.representation(position)
            processor = self.__class__(method=self.method)
            if planned is None:
                processor.fit(before, generator=generator)
            else:
                processor._indices = tuple(planned[position])
                processor.permutation = torch.tensor(
                    processor._indices,
                    dtype=torch.long,
                    device=before.device,
                )
                processor._fitted = True

            key = (
                table._member_locations[position],
                processor._indices,
            )
            processor_index = keys.get(key)
            if processor_index is None:
                processor_index = len(representations)
                keys[key] = processor_index
                self.processors.append(processor)
                positions.append(position)
                representations.append(processor.transform(before))
            member_to_processor.append(processor_index)

        self._member_to_processor = tuple(member_to_processor)
        self._processor_positions = tuple(positions)
        return EnsembleTable.pack(
            representations=representations,
            member_representation_ids=self._member_to_processor,
            member_ids=table._member_ids,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Apply the fitted member permutations."""
        representations = tuple(
            cast(ShuffleColumns, processor).transform(
                table.representation(position)
            )
            for processor, position in zip(
                self.processors,
                self._processor_positions,
            )
        )
        return EnsembleTable.pack(
            representations=representations,
            member_representation_ids=self._member_to_processor,
            member_ids=table._member_ids,
        )

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Invert each output with its fitted member permutation."""
        representations = tuple(
            cast(
                ShuffleColumns,
                self.processors[processor_index],
            ).inverse_transform(table.representation(position))
            for position, processor_index in enumerate(
                self._member_to_processor
            )
        )
        return EnsembleTable.pack(
            representations=representations,
            member_representation_ids=tuple(range(table.num_members)),
            member_ids=table._member_ids,
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
