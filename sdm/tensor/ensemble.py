from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    r"""Table data for multiple members of a model ensemble.

    Each ensemble member receives a :class:`~sdm.tensor.TableTensor` as input.
    Members may share the same table or hold member-specific tables produced by
    per-member preprocessing. Shared tables are stored once.

    Compatible member tables are batched into groups along a leading dimension
    for joint processing. Two tables are compatible when they have the same
    shape, column schema, block layout, device, and categorical vocabularies.

    .. testcode::

        import torch
        from sdm import EnsembleTable, TableTensor

        original = TableTensor.from_tensor(
            torch.tensor([[1.0], [2.0]])
        )
        normalized = TableTensor.from_tensor(
            torch.tensor([[-1.0], [1.0]])
        )

        ensemble = EnsembleTable.from_member_tables(
            tables=(original, normalized),
            member_table_ids=(0, 1, 0, 1),
        )

        assert ensemble.num_members == 4
        assert ensemble.member_table(0).equal(original)
        assert ensemble.member_table(1).equal(normalized)

    Args:
        table: A :class:`~sdm.tensor.TableTensor` shared by all ensemble
            members.
        num_members: Number of ensemble members.
    """

    def __init__(self, table: TableTensor, *, num_members: int) -> None:
        if num_members <= 0:
            raise ValueError("Expected 'num_members' to be positive.")
        self._member_groups = (cast(TableTensor, table.unsqueeze(0)),)
        self._member_locations = ((0, 0),) * num_members

    @classmethod
    def from_member_tables(
        cls,
        tables: Sequence[TableTensor],
        member_table_ids: Sequence[int],
    ) -> Self:
        r"""Create an ensemble table from member-specific tables.

        ``member_table_ids[i]`` selects the table used by member ``i``.
        Reusing an ID means that members share the same table. Compatible
        tables are automatically batched into groups for joint processing.

        Categorical tables are grouped only when they reference the same
        category vocabulary objects.

        Args:
            tables: Distinct tables that members may reference.
            member_table_ids: For each member, the index of its table in
                ``tables``.

        Returns:
            An ensemble table preserving member order.
        """
        if len(tables) == 0:
            raise ValueError("Expected at least one table.")
        if len(member_table_ids) == 0:
            raise ValueError("Expected at least one ensemble member.")
        if any(
            table_id < 0 or table_id >= len(tables)
            for table_id in member_table_ids
        ):
            raise ValueError(
                "'member_table_ids' references an unknown table."
            )

        compatible_groups: dict[tuple[object, ...], list[int]] = {}
        for index, table in enumerate(tables):
            # Shape, schema, block layout, device, and categorical vocabularies
            # must match for torch.stack to preserve member semantics.
            compatibility_key = (
                tuple(table.size()),
                tuple(
                    (stype, columns)
                    for stype, columns in table.columns.items()
                ),
                tuple(
                    (stype, type(block), block.dtype)
                    for stype, block in table.items()
                ),
                table.device,
                tuple(
                    id(category)
                    for category in table.categorical.categories
                ),
            )
            compatible_groups.setdefault(compatibility_key, []).append(index)

        member_groups: list[TableTensor] = []
        input_locations: dict[int, tuple[int, int]] = {}
        for indices in compatible_groups.values():
            group_index = len(member_groups)
            member_groups.append(
                cast(TableTensor, tables[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        [tables[index] for index in indices],
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
        ensemble._member_groups = tuple(member_groups)
        ensemble._member_locations = tuple(
            input_locations[index] for index in member_table_ids
        )
        return ensemble

    @property
    def num_members(self) -> int:
        """Return the number of ensemble members."""
        return len(self._member_locations)

    def member_table(self, member_id: int) -> TableTensor:
        """Return the table for one ensemble member.

        Args:
            member_id: Zero-based member index.
        """
        group_index, position = self._member_locations[member_id]
        return self._member_groups[group_index][position]

    def member_groups(self) -> Iterator[TableTensor]:
        """Yield groups of compatible member tables.

        Each group is a :class:`~sdm.tensor.TableTensor` with a leading
        dimension of size ``G``, where ``G`` is the number of members in the
        group. Use this to process all members in a group jointly as a batch.
        """
        return iter(self._member_groups)

    def __repr__(self) -> str:
        num_member_tables = sum(g.size(0) for g in self._member_groups)
        return (
            f"{self.__class__.__name__}(num_members={self.num_members}, "
            f"num_member_tables={num_member_tables})"
        )
