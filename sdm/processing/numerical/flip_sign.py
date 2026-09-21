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
        signs = []
        for group_id in range(ensemble_table.num_groups):
            group = ensemble_table.expanded_group(group_id)
            sign = group.numerical.new_empty(
                (*group.size()[:-2], 1, group.numerical.size(-1))
            )
            sign.bernoulli_(self.probability, generator=generator)
            sign.mul_(-2).add_(1)
            signs.append(sign)
        self._signs = BufferList(signs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:

        groups = []
        for group_id in range(ensemble_table.num_groups):
            group = ensemble_table.expanded_group(group_id)
            sign = self._signs[group_id]
            groups.append(
                group.replace_blocks(numerical=group.numerical * sign)
            )

        locations = []
        next_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, next_position[group_id]))
            next_position[group_id] += 1

        return EnsembleTable(groups, locations)

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
