# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor


class RandomProjection(EnsembleProcessor):
    r"""Randomly project numerical columns.

    Each logical ensemble member retains its fitted projection when table
    groups change. Transforming requires the fitted number of members.

    Args:
        channels: Number of output numerical columns.
        init: The projection weight initialization.
            ``"normal"`` samples Gaussian weights scaled by
            ``1 / sqrt(channels)``.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        channels: int,
        *,
        init: Literal["normal"] = "normal",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.init = init
        self._weights: BufferList[torch.Tensor] = BufferList()
        self._weight_locations: tuple[tuple[int, int], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, int], ...]:  # noqa: D102
        return self._weight_locations

    def set_extra_state(self, state: object) -> None:  # noqa: D102
        self._weight_locations = cast(tuple[tuple[int, int], ...], state)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        weights = []
        for i, group in enumerate(ensemble_table):
            num_members = ensemble_table.num_members_in_group(i)
            weight = group.numerical.new_empty(
                (
                    num_members,
                    *group.size()[1:-2],
                    self.channels,
                    group.numerical.size(-1),
                )
            )
            assert self.init == "normal"
            weight.normal_(std=self.channels**-0.5, generator=generator)
            weights.append(weight)
        self._weights = BufferList(weights)
        locations = []
        group_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, group_position[group_id]))
            group_position[group_id] += 1
        self._weight_locations = tuple(locations)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if ensemble_table.num_members != len(self._weight_locations):
            raise RuntimeError(
                f"RandomProjection was fitted with "
                f"{len(self._weight_locations)} ensemble members, but got "
                f"{ensemble_table.num_members}"
            )

        weight_locations: list[list[tuple[int, int]]] = [
            [] for _ in range(ensemble_table.num_groups)
        ]
        for (group_id, _), location in zip(
            ensemble_table._locations, self._weight_locations, strict=True
        ):
            weight_locations[group_id].append(location)

        groups = []
        for i in range(ensemble_table.num_groups):
            group = ensemble_table.expanded_group(i)
            locations = weight_locations[i]
            if locations:
                fitted_group = locations[0][0]
                weight = self._weights[fitted_group]
                if locations != [
                    (fitted_group, position)
                    for position in range(weight.size(0))
                ]:
                    weight = torch.stack(
                        [
                            self._weights[g][position]
                            for g, position in locations
                        ]
                    )
                numerical = group.numerical @ weight.transpose(-1, -2)
            else:
                numerical = group.numerical.new_empty(
                    (*group.size()[:-1], self.channels)
                )
            projected = TableTensor(
                columns={
                    Stype.numerical: [f"rp_{i}" for i in range(self.channels)]
                },
                numerical=numerical,
            )
            group = cast(
                TableTensor,
                torch.cat(
                    [group.drop_stypes(Stype.numerical), projected], dim=-1
                ),
            )
            groups.append(group)

        locations = []
        group_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, group_position[group_id]))
            group_position[group_id] += 1

        return EnsembleTable(groups=groups, locations=locations)
