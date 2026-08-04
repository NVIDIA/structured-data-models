from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    r"""Container for the tabular inputs of a model ensemble.

    A :class:`~sdm.tensor.TableTensor` stores one tensorized table. An
    :class:`EnsembleTable` associates each member position with one such table.
    Multiple members may reference the same table, which is stored only once.

    For processing, distinct compatible tables are stacked along a new leading
    dimension. Tables form separate groups when their shape, column schema,
    block layout, device, or categorical vocabularies differ. Model code can
    access one member's table with :meth:`member_table`, while processor code
    can iterate over the stacked groups with :meth:`member_groups`.

    .. testcode::

        import torch
        from sdm import EnsembleTable, TableTensor

        original = TableTensor.from_tensor(
            torch.tensor([[1.0], [2.0]]),
            columns=("value",),
        )
        normalized = TableTensor.from_tensor(
            torch.tensor([[-1.0], [1.0]]),
            columns=("value",),
        )
        selected = TableTensor.from_tensor(
            torch.tensor([[10.0], [20.0]]),
            columns=("selected_value",),
        )

        ensemble = EnsembleTable.from_member_tables(
            tables=(original, normalized, selected),
            member_table_ids=(0, 1, 2, 0),
        )

        # Model code accesses tables in member order.
        assert ensemble.num_members == 4
        assert ensemble.member_table(0).equal(original)
        assert ensemble.member_table(3).equal(original)

        # Processor code receives two groups of compatible tables.
        groups = tuple(ensemble.member_groups())
        assert len(groups) == 2
        assert groups[0].size() == (2, 2, 1)
        assert groups[1].size() == (1, 2, 1)

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

        Each entry in ``tables`` is stored once. ``member_table_ids[i]``
        selects the table associated with member position ``i``. Reusing an
        index makes multiple members reference the same stored table.

        Compatible tables are stacked into groups for joint processing;
        incompatible tables remain in separate groups. Categorical tables are
        compatible only when they reference the same category vocabularies.

        Args:
            tables: Tables available to the ensemble members.
            member_table_ids: Index into ``tables`` for each member position.

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
            raise ValueError("'member_table_ids' references an unknown table.")

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
                    id(category) for category in table.categorical.categories
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
        """Return the table associated with one member for model execution.

        Args:
            member_id: Zero-based member index.
        """
        group_index, position = self._member_locations[member_id]
        return self._member_groups[group_index][position]

    def member_groups(self) -> Iterator[TableTensor]:
        """Yield the stored groups of compatible tables for processing.

        Each group is a :class:`~sdm.tensor.TableTensor` with a leading
        dimension of size ``G``, where ``G`` is the number of distinct tables
        in the group, not the number of members referencing them. Tables in a
        group can be processed jointly.
        """
        return iter(self._member_groups)

    def __repr__(self) -> str:
        num_member_tables = sum(g.size(0) for g in self._member_groups)
        return (
            f"{self.__class__.__name__}(num_members={self.num_members}, "
            f"num_member_tables={num_member_tables})"
        )
