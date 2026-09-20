# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype
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

        member_ids_by_group: list[list[int]] = [
            [] for _ in range(ensemble_table.num_groups)
        ]
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            member_ids_by_group[group_id].append(member_id)

        groups = []
        locations = list(ensemble_table._locations)
        for group_id, (group, member_ids) in enumerate(
            zip(
                ensemble_table._iter_groups(),
                member_ids_by_group,
                strict=True,
            )
        ):
            positions = [
                ensemble_table._locations[member_id][1]
                for member_id in member_ids
            ]
            if len(set(positions)) == len(positions):
                signs = group.numerical.new_ones(
                    (*group.numerical.size()[:-2], 1, group.numerical.size(-1))
                )
                for member_id, position in zip(
                    member_ids, positions, strict=True
                ):
                    signs[position] = self._signs[member_id]
            else:
                group = ensemble_table.expanded_group(group_id)
                signs = torch.stack(
                    [self._signs[member_id] for member_id in member_ids]
                )
                for position, member_id in enumerate(member_ids):
                    locations[member_id] = (group_id, position)

            groups.append(
                group.replace_blocks(numerical=group.numerical * signs)
            )
        return EnsembleTable(groups=groups, locations=locations)

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
