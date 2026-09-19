# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class FlipSign(EnsembleProcessor, EnsembleInvertibleMixin):
    """Randomly negate numerical columns.

    Args:
        probability: Probability of negating each numerical column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.probability = probability
        self._signs: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        # TODO: Vectorize sign draws and application over ensemble members.
        signs = []
        for member_id in range(len(ensemble_table)):
            numerical = ensemble_table[member_id].numerical
            sign = numerical.new_empty(
                (*numerical.size()[:-2], 1, numerical.size(-1))
            )
            sign.bernoulli_(self.probability, generator=generator)
            sign.mul_(-2).add_(1)
            signs.append(sign)
        self._signs = BufferList(signs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._signs) != len(ensemble_table):
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._signs)} ensemble members, but got "
                f"{len(ensemble_table)}."
            )

        # Reuse the single group when positions biject onto its rows: keeps
        # a stable layout for position-dependent fitted processors (e.g.
        # `Standardize`) placed around this one in a `Sequential`.
        if ensemble_table.num_groups == 1:
            group = next(ensemble_table._iter_groups())
            positions = tuple(
                position for _, position in ensemble_table._locations
            )
            if sorted(positions) == list(range(group.size(0))):
                member_by_position = {
                    position: member_id
                    for member_id, position in enumerate(positions)
                }
                signs = torch.stack(
                    [
                        self._signs[member_by_position[position]]
                        for position in range(group.size(0))
                    ],
                    dim=0,
                )
                new_group = group.replace_blocks(
                    numerical=group.numerical * signs
                )
                return ensemble_table.replace_groups([new_group])

        # Fallback: members sharing a position diverge after negation, so
        # each member becomes its own group.
        tables: list[TableTensor] = []
        for member_id in range(len(ensemble_table)):
            table = ensemble_table[member_id]
            tables.append(
                table.replace_blocks(
                    numerical=table.numerical * self._signs[member_id]
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._transform_ensemble(ensemble_table)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(probability={self.probability})"
        )
