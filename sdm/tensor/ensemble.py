from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    """Store table representations for an ensemble.

    Members can share a representation. Representations with matching shape,
    schema, block types and dtypes, device, and categorical vocabulary objects
    are stacked along a leading dimension for joint processing.

    Args:
        table: Table with shape ``[..., R, C]`` shared by all ensemble members.
        num_members: Number of ensemble members.
    """

    def __init__(self, table: TableTensor, *, num_members: int) -> None:
        self._packed_representations = (cast(TableTensor, table.unsqueeze(0)),)
        self._member_locations = ((0, 0),) * num_members

    @classmethod
    def from_representations(
        cls,
        representations: Sequence[TableTensor],
        member_representation_ids: Sequence[int],
    ) -> Self:
        """Create an ensemble table from member representations.

        Compatible representations are stacked in input order.

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
