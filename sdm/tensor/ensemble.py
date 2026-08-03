from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    r"""Table data for multiple members of a model ensemble.

    A model ensemble combines multiple estimators, called members.
    :class:`EnsembleTable` maps each member to the
    :class:`~sdm.tensor.TableTensor` it uses. Members may share the same table
    or use differently processed representations. Shared data is stored once.
    Compatible representations are stacked for joint processing, while
    incompatible representations remain separate.

    .. testcode::

        import torch
        from sdm import EnsembleTable, TableTensor

        original = TableTensor.from_tensor(
            torch.tensor([[1.0], [2.0]])
        )
        normalized = TableTensor.from_tensor(
            torch.tensor([[-1.0], [1.0]])
        )

        ensemble = EnsembleTable.from_representations(
            representations=(original, normalized),
            member_representation_ids=(0, 1, 0, 1),
        )

        assert ensemble.num_members == 4
        assert ensemble.representation(0).equal(original)
        assert ensemble.representation(1).equal(normalized)

    Args:
        table: A :class:`TableTensor` shared by all ensemble members.
        num_members: Number of ensemble members.
    """

    def __init__(self, table: TableTensor, *, num_members: int) -> None:
        if num_members <= 0:
            raise ValueError("Expected 'num_members' to be positive.")
        self._packed_representations = (cast(TableTensor, table.unsqueeze(0)),)
        self._member_locations = ((0, 0),) * num_members

    @classmethod
    def from_representations(
        cls,
        representations: Sequence[TableTensor],
        member_representation_ids: Sequence[int],
    ) -> Self:
        r"""Create an ensemble table from member-specific representations.

        ``member_representation_ids[i]`` selects the representation used by
        member ``i``. Reusing an ID means that members share the same
        representation. Compatible representations are stacked without
        changing member order.

        Categorical representations are stacked only when they reference the
        same category vocabulary objects.

        Args:
            representations: Table representations that members may reference.
            member_representation_ids: For each member, its index into
                ``representations``.

        Returns:
            An ensemble table preserving member order.
        """
        if len(representations) == 0:
            raise ValueError("Expected at least one representation.")
        if len(member_representation_ids) == 0:
            raise ValueError("Expected at least one ensemble member.")
        if any(
            representation_id < 0 or representation_id >= len(representations)
            for representation_id in member_representation_ids
        ):
            raise ValueError(
                "'member_representation_ids' references an unknown "
                "representation."
            )

        compatible_groups: dict[tuple[object, ...], list[int]] = {}
        for index, representation in enumerate(representations):
            # Shape, schema, block layout, device, and categorical vocabularies
            # must match for torch.stack to preserve member semantics.
            compatibility_key = (
                tuple(representation.size()),
                tuple(
                    (stype, columns)
                    for stype, columns in representation.columns.items()
                ),
                tuple(
                    (stype, type(block), block.dtype)
                    for stype, block in representation.items()
                ),
                representation.device,
                tuple(
                    id(category)
                    for category in representation.categorical.categories
                ),
            )
            compatible_groups.setdefault(compatibility_key, []).append(index)

        packed_representations: list[TableTensor] = []
        input_locations: dict[int, tuple[int, int]] = {}
        for indices in compatible_groups.values():
            packed_index = len(packed_representations)
            packed_representations.append(
                cast(TableTensor, representations[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        [representations[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            input_locations.update(
                {
                    input_index: (packed_index, representation_index)
                    for representation_index, input_index in enumerate(indices)
                }
            )

        table = cls.__new__(cls)
        table._packed_representations = tuple(packed_representations)
        table._member_locations = tuple(
            input_locations[index] for index in member_representation_ids
        )
        return table

    @property
    def num_members(self) -> int:
        """Return the number of ensemble members."""
        return len(self._member_locations)

    def representation(self, member_id: int) -> TableTensor:
        """Return the table representation used by one member.

        Args:
            member_id: Zero-based ensemble member index.
        """
        packed_index, representation_index = self._member_locations[member_id]
        return self._packed_representations[packed_index][representation_index]

    def iter_packed_representations(self) -> Iterator[TableTensor]:
        """Yield compatible representations.

        Representations are stacked along a leading dimension.
        """
        return iter(self._packed_representations)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_members={self.num_members}, "
            f"num_representations="
            f"{sum(table.size(0) for table in self._packed_representations)})"
        )
