# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor


class FlipSign(EnsembleProcessor, EnsembleInvertibleMixin):
    """Randomly negate numerical columns.

    Every ensemble member draws its own independent signs, so members
    sharing one input table receive distinct tables. A single table draws
    one sign vector per leading batch element. Pass ``generator`` to
    ``fit()`` to make the signs reproducible.

    Args:
        probability: Probability of negating each numerical column.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True
    _empty_sign: Tensor

    def __init__(
        self,
        probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.probability = probability
        self.register_buffer("_empty_sign", torch.empty(0), persistent=False)
        self._signs: BufferList[Tensor] = BufferList()

    @property
    def sign(self) -> Tensor:
        """Signs of the first ensemble member, or empty before fitting."""
        if len(self._signs) == 0:
            return self._empty_sign
        return self._signs[0]

    @staticmethod
    def _member_ids_by_group(
        ensemble_table: EnsembleTable,
    ) -> list[list[int]]:
        # TODO: Add to EnsembleTable directly.
        member_ids_by_group: list[list[int]] = [
            [] for _ in range(ensemble_table.num_groups)
        ]
        for member_id, (group_id, _) in enumerate(ensemble_table._locations):
            member_ids_by_group[group_id].append(member_id)
        return member_ids_by_group

    def _check_num_members(self, ensemble_table: EnsembleTable) -> None:
        if len(self._signs) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._signs)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        signs: list[Tensor] = [torch.empty(0)] * ensemble_table.num_members
        for group, member_ids in zip(
            ensemble_table,
            self._member_ids_by_group(ensemble_table),
            strict=True,
        ):
            numerical = group.numerical  # [positions, *batch, rows, columns]
            sign = torch.empty(
                (
                    len(member_ids),
                    *numerical.size()[1:-2],
                    1,
                    numerical.size(-1),
                ),
                dtype=numerical.dtype,
                device=(
                    numerical.device if generator is None else generator.device
                ),
            )
            sign.bernoulli_(self.probability, generator=generator)
            sign.mul_(-2).add_(1)
            for member_id, member_sign in zip(
                member_ids,
                sign.to(numerical.device).unbind(0),
                strict=True,
            ):
                signs[member_id] = member_sign
        self._signs = BufferList(signs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        self._check_num_members(ensemble_table)
        groups: list[TableTensor] = []
        locations = [(0, 0)] * ensemble_table.num_members
        for group_id, member_ids in enumerate(
            self._member_ids_by_group(ensemble_table)
        ):
            group = ensemble_table.expanded_group(group_id)
            sign = torch.stack(
                [self._signs[member_id] for member_id in member_ids]
            )  # [members, *batch, 1, columns]
            groups.append(
                group.replace_blocks(numerical=group.numerical * sign)
            )
            for position, member_id in enumerate(member_ids):
                locations[member_id] = (group_id, position)
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
