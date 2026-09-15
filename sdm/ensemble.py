# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Self, cast

import torch
from torch import Tensor

from sdm import Stype, StypeLike
from sdm.tensor import TableTensor
from sdm.tensor.mixin import DeviceMixin


class EnsembleTable(DeviceMixin):
    """Store and group input tables for an ensemble.

    Each ensemble member is associated with one table. Shared tables are stored
    only once. Compatible tables are stacked along a leading dimension and
    form a group, so processors can process them together. Incompatible tables
    remain in separate groups.

    Use :meth:`table` to access a member's table. Iterate over the
    :class:`EnsembleTable` to process its groups, and :meth:`replace_groups`
    to build an ensemble table from the processed groups.

    .. testcode::

        import torch
        from sdm import EnsembleTable, TableTensor

        estimator_table1 = TableTensor.from_tensor(
            tensor=torch.tensor([[1.0], [2.0]]),
            columns=("value",),
        )
        estimator_table2 = TableTensor.from_tensor(
            tensor=torch.tensor([[-1.0], [1.0]]),
            columns=("value",),
        )
        estimator_table3 = TableTensor.from_tensor(
            tensor=torch.tensor([[10.0], [20.0]]),
            columns=("selected_value",),
        )

        ensemble = EnsembleTable.from_tables(
            tables=(estimator_table1, estimator_table2, estimator_table3),
            member_table_ids=(0, 1, 2, 0),
        )

        # Access tables in member order.
        assert ensemble.num_members == 4
        assert ensemble.table(0).equal(estimator_table1)
        assert ensemble.table(3).equal(estimator_table1)

        # Iterate over two groups of compatible tables.
        groups = tuple(ensemble)
        assert len(groups) == 2
        assert groups[0].size() == (2, 2, 1)
        assert groups[1].size() == (1, 2, 1)

    Args:
        groups: Sequence of
            :class:`~sdm.tensor.TableTensor`, optionally stacked along their
            leading dimension.
        locations: ``(group, batch)`` location of each ensemble member.
    """

    _groups: tuple[TableTensor, ...]
    # Group index and position within that group, per ensemble member.
    _locations: tuple[tuple[int, int], ...]

    def __init__(
        self,
        groups: Sequence[TableTensor],
        locations: Sequence[tuple[int, int]],
    ) -> None:
        self._groups = tuple(groups)
        self._locations = tuple(locations)

    @classmethod
    def from_table(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> Self:
        """Create an ensemble, sharing one table across all members.

        Args:
            table: Table used across members.
            num_members: Number of ensemble members.

        Returns:
            An :class:`~sdm.EnsembleTable` with one group.
        """
        return cls(
            groups=(cast(TableTensor, table.unsqueeze(0)),),
            locations=((0, 0),) * num_members,
        )

    @classmethod
    def from_tables(
        cls,
        tables: Sequence[TableTensor],
        member_table_ids: Sequence[int],
    ) -> Self:
        """Create an ensemble table from tables and their member assignments.

        ``member_table_ids`` contains one table index per member. For
        example, ``(0, 1, 0)`` assigns the first table to members 0 and 2 and
        the second table to member 1.

        Args:
            tables: Tables available to the ensemble members.
            member_table_ids: Index into ``tables`` for each ensemble member.

        Returns:
            An ensemble table preserving member order.
        """
        referenced_table_ids = set(member_table_ids)
        compatible_groups: dict[tuple[object, ...], list[int]] = {}
        for index, table in enumerate(tables):
            if index not in referenced_table_ids:
                continue
            # Shape, schema, block layout, device, and categorical vocabularies
            # must match for torch.stack to preserve member semantics.
            compatibility_key = (
                tuple(
                    (stype, columns)
                    for stype, columns in table.columns.items()
                ),
                tuple(
                    (
                        stype,
                        type(block),
                        block.size(),
                        block.layout,
                        block.dtype,
                    )
                    for stype, block in table.items()
                ),
                table.device,
                tuple(
                    id(category) for category in table.categorical.categories
                ),
            )
            compatible_groups.setdefault(compatibility_key, []).append(index)

        groups: list[TableTensor] = []
        input_locations: dict[int, tuple[int, int]] = {}
        for indices in compatible_groups.values():
            group_index = len(groups)
            groups.append(
                cast(TableTensor, tables[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        tensors=[tables[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            input_locations.update(
                {
                    input_index: (group_index, position)
                    for position, input_index in enumerate(indices)
                }
            )

        return cls(
            groups=groups,
            locations=[input_locations[index] for index in member_table_ids],
        )

    def select_members(self, member_ids: Sequence[int]) -> Self:
        """Return the selected ensemble members in the requested order.

        Args:
            member_ids: Logical member positions to select.

        Returns:
            An ensemble table containing the selected members.
        """
        member_ids = tuple(member_ids)
        if member_ids == tuple(range(self.num_members)):
            return self

        locations = tuple(
            self._locations[member_id] for member_id in member_ids
        )
        positions_by_group: dict[int, dict[int, int]] = {}
        for group_id, position in locations:
            positions = positions_by_group.setdefault(group_id, {})
            positions.setdefault(position, len(positions))

        groups: list[TableTensor] = []
        group_ids: dict[int, int] = {}
        for new_group_id, (group_id, positions) in enumerate(
            positions_by_group.items()
        ):
            group = self._groups[group_id]
            selected_positions = tuple(positions)
            group_ids[group_id] = new_group_id
            if selected_positions == tuple(range(group.size(0))):
                groups.append(group)
            elif len(selected_positions) == 1:
                groups.append(
                    cast(
                        TableTensor,
                        group.narrow(0, selected_positions[0], 1),
                    )
                )
            else:
                groups.append(
                    cast(
                        TableTensor,
                        torch.stack(
                            [
                                group[position]
                                for position in selected_positions
                            ],
                            dim=0,
                        ),
                    )
                )

        ensemble = copy.copy(self)
        ensemble._groups = tuple(groups)
        ensemble._locations = tuple(
            (group_ids[group_id], positions_by_group[group_id][position])
            for group_id, position in locations
        )
        return ensemble

    @classmethod
    def gather_members(
        cls,
        tables: Sequence[Self],
        member_ids: Sequence[int],
    ) -> Self:
        """Gather members from multiple ensemble tables.

        ``tables[i].table(member_ids[i])`` supplies output member ``i``.

        Args:
            tables: Source ensemble table for each output member.
            member_ids: Logical source member position for each output member.

        Returns:
            An ensemble table preserving logical member order.
        """
        if len(tables) != len(member_ids):
            raise ValueError("Expected one source member per ensemble table")

        first = tables[0]
        if all(table is first for table in tables[1:]):
            return first.select_members(member_ids)

        # TODO: This path occurs when members come from multiple sources, such
        # as different Choice options. It can be optimized by refining their
        # compatible group layouts and gathering their rows directly into the
        # output groups.
        outputs: list[TableTensor] = []
        output_id_by_source: dict[tuple[int, tuple[int, int]], int] = {}
        member_table_ids = []
        for table, member_id in zip(tables, member_ids, strict=True):
            location = table._locations[member_id]
            key = (id(table), location)
            output_id = output_id_by_source.get(key)
            if output_id is None:
                output_id = len(outputs)
                output_id_by_source[key] = output_id
                outputs.append(table.table(member_id))
            member_table_ids.append(output_id)

        return cls.from_tables(
            tables=outputs,
            member_table_ids=member_table_ids,
        )

    @property
    def num_members(self) -> int:
        """Return the number of ensemble members."""
        return len(self._locations)

    @property
    def num_groups(self) -> int:
        """Return the number of groups of compatible tables."""
        return len(self._groups)

    def num_members_in_group(self, group_id: int) -> int:
        """Return the number of members of a group.

        Args:
            group_id: Zero-based group index.
        """
        return sum(i == group_id for i, _ in self._locations)

    def table(self, member_id: int) -> TableTensor:
        """Return the table associated with one ensemble member.

        Args:
            member_id: Zero-based member index.
        """
        group_index, position = self._locations[member_id]
        return self._groups[group_index][position]

    def expanded_group(self, group_id: int) -> TableTensor:
        """Return the logical members assigned to one group.

        Args:
            group_id: Zero-based group index.
        """
        group = self._groups[group_id]
        positions = tuple(
            position for i, position in self._locations if i == group_id
        )
        if positions == tuple(range(group.size(0))):
            return group
        if group.size(0) == 1:
            return cast(
                TableTensor,
                group.expand(len(positions), *group.size()[1:]),
            )
        index = torch.tensor(positions, device=group.device)
        return cast(TableTensor, group.index_select(0, index))

    def __iter__(self) -> Iterator[TableTensor]:
        """Iterate over groups of compatible tables."""
        return iter(self._groups)

    def _tensors(self) -> Iterator[Tensor]:
        yield from self._groups

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.replace_groups(
            [cast(TableTensor, fn(group)) for group in self._groups]
        )

    def select_stypes(
        self,
        stypes: StypeLike | Iterable[StypeLike],
    ) -> Self:
        r"""Return an ensemble table containing only ``stypes`` columns.

        Args:
            stypes: The semantic type or semantic types to select.

        Returns:
            An ensemble table preserving its logical member assignment.
        """
        if isinstance(stypes, (str, Stype)):
            stypes = (stypes,)
        stypes = tuple(Stype(stype) for stype in stypes)

        if len(stypes) == 0:
            return self.replace_groups(
                [group.select_columns(()) for group in self]
            )
        return self.replace_groups(
            [group.select_stypes(stypes) for group in self]
        )

    @classmethod
    def concatenate_columns(cls, tables: Sequence[Self]) -> Self:
        r"""Concatenate ensemble tables column-wise by logical member.

        Args:
            tables: Ensemble tables with the same number of logical members.

        Returns:
            An ensemble table preserving logical member order.
        """
        if len(tables) == 0:
            raise ValueError("Expected at least one ensemble table")

        first = tables[0]
        if any(table.num_members != first.num_members for table in tables[1:]):
            raise ValueError(
                "Cannot concatenate ensemble tables with different member "
                "counts"
            )

        nonempty_tables = tuple(
            table
            for table in tables
            if any(group.size(-1) > 0 for group in table)
        )
        if len(nonempty_tables) > 0:
            tables = nonempty_tables
            first = tables[0]
        if len(tables) == 1:
            return first

        if all(table._locations == first._locations for table in tables[1:]):
            return first.replace_groups(
                [
                    cast(TableTensor, torch.cat(groups, dim=-1))
                    for groups in zip(*tables, strict=True)
                ]
            )

        # TODO: Concatenate compatible groups directly and unpack logical
        # members only when their layouts differ.
        outputs: list[TableTensor] = []
        output_id_by_locations: dict[tuple[tuple[int, int], ...], int] = {}
        member_table_ids = []
        for member_id in range(first.num_members):
            locations = tuple(table._locations[member_id] for table in tables)
            output_id = output_id_by_locations.get(locations)
            if output_id is None:
                output_id = len(outputs)
                output_id_by_locations[locations] = output_id
                outputs.append(
                    cast(
                        TableTensor,
                        torch.cat(
                            tuple(table.table(member_id) for table in tables),
                            dim=-1,
                        ),
                    )
                )
            member_table_ids.append(output_id)

        return cls.from_tables(
            tables=outputs,
            member_table_ids=member_table_ids,
        )

    def replace_groups(self, groups: Sequence[TableTensor]) -> Self:
        """Return an ensemble table with its groups replaced.

        Args:
            groups: One replacement group per current group.

        Returns:
            An ensemble table over ``groups`` with the current member
            assignment.
        """
        if len(groups) != self.num_groups:
            raise ValueError(
                f"Expected one replacement per group of compatible tables "
                f"({self.num_groups}), got {len(groups)}."
            )
        ensemble = copy.copy(self)
        ensemble._groups = tuple(groups)
        return ensemble

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_members={self.num_members}, num_groups={self.num_groups})"
        )
