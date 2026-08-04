from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    """Store and group input tables for an ensemble.

    Each ensemble member is associated with one table. Shared tables are stored
    only once. Distinct tables with compatible schemas are stacked so
    processors can process them together; incompatible tables remain separate.

    Use :meth:`table` to access a member's table. Iterate over the
    :class:`EnsembleTable` to process its compatible groups.

    .. testcode::

        import torch
        from sdm.tensor import EnsembleTable, TableTensor

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
        table: A :class:`~sdm.tensor.TableTensor` shared by all ensemble
            members.
        num_members: Number of ensemble members.
    """

    def __init__(self, table: TableTensor, *, num_members: int) -> None:
        self._groups = (cast(TableTensor, table.unsqueeze(0)),)
        self._member_locations = ((0, 0),) * num_members

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

        ensemble = cls.__new__(cls)
        ensemble._groups = tuple(groups)
        ensemble._member_locations = tuple(
            input_locations[index] for index in member_table_ids
        )
        return ensemble

    @property
    def num_members(self) -> int:
        """Return the number of ensemble members."""
        return len(self._member_locations)

    def table(self, member_id: int) -> TableTensor:
        """Return the table associated with one ensemble member.

        Args:
            member_id: Zero-based member index.
        """
        group_index, position = self._member_locations[member_id]
        return self._groups[group_index][position]

    def __iter__(self) -> Iterator[TableTensor]:
        """Iterate over groups of compatible tables."""
        return iter(self._groups)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_members={self.num_members}, "
            f"num_groups={len(self._groups)})"
        )
