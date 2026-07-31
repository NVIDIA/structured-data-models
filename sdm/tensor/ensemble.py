from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


class EnsembleTable:
    r"""Store shared and distinct representations of one table.

    Args:
        table: The initial shared table with shape ``[..., R, C]``.
        num_members: Number of logical ensemble members.
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
        r"""Create an ensemble table from member representations.

        Args:
            representations: Distinct results in provenance order.
            member_representation_ids: Representation index for every member.
        """
        representations = tuple(representations)
        member_representation_ids = tuple(member_representation_ids)
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

        buckets: dict[tuple[object, ...], list[int]] = {}
        for index, representation in enumerate(representations):
            key = (
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
            buckets.setdefault(key, []).append(index)

        packed_representations: list[TableTensor] = []
        input_locations: dict[int, tuple[int, int]] = {}
        for indices in buckets.values():
            packed_index = len(packed_representations)
            packed_representations.append(
                cast(TableTensor, representations[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        tuple(representations[index] for index in indices),
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
        """Return the number of logical ensemble members."""
        return len(self._member_locations)

    def representation(self, member_id: int) -> TableTensor:
        r"""Return the table representation used by one member.

        Args:
            member_id: Stable member position.
        """
        packed_index, representation_index = self._member_locations[member_id]
        return self._packed_representations[packed_index][representation_index]

    def iter_packed_representations(self) -> Iterator[TableTensor]:
        """Iterate over schema-compatible packed representations."""
        return iter(self._packed_representations)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_members={self.num_members}, "
            f"num_representations="
            f"{sum(table.size(0) for table in self._packed_representations)})"
        )
