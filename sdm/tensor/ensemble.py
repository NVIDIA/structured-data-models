from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from typing_extensions import Self

from sdm.tensor.table import TableTensor


def _representation_key(table: TableTensor) -> tuple[object, ...]:
    return (
        tuple(table.size()),
        tuple((stype, columns) for stype, columns in table.columns.items()),
        tuple(
            (stype, type(block), block.dtype) for stype, block in table.items()
        ),
        table.device,
        tuple(id(category) for category in table.categorical.categories),
    )


@dataclass(frozen=True, init=False)
class EnsembleTable:
    r"""Store shared and distinct representations of one table.

    Args:
        table: The initial shared table with shape ``[..., R, C]``.
        num_members: Positive number of logical ensemble members.
    """

    _packed_representations: tuple[TableTensor, ...]
    _member_locations: tuple[tuple[int, int], ...]

    def __init__(self, table: TableTensor, *, num_members: int) -> None:
        if num_members < 1:
            raise ValueError("'num_members' needs to be positive.")
        object.__setattr__(
            self,
            "_packed_representations",
            (cast(TableTensor, table.unsqueeze(0)),),
        )
        object.__setattr__(
            self,
            "_member_locations",
            ((0, 0),) * num_members,
        )

    @classmethod
    def _from_packed_representations(
        cls,
        packed_representations: tuple[TableTensor, ...],
        member_locations: tuple[tuple[int, int], ...],
    ) -> Self:
        table = cls.__new__(cls)
        object.__setattr__(
            table, "_packed_representations", packed_representations
        )
        object.__setattr__(table, "_member_locations", member_locations)
        return table

    @classmethod
    def pack(
        cls,
        representations: Sequence[TableTensor],
        member_representation_ids: Sequence[int],
    ) -> Self:
        r"""Pack representations by compatible metadata.

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
        if (
            len({representation.device for representation in representations})
            > 1
        ):
            raise ValueError(
                "Expected all ensemble representations on the same device."
            )

        buckets: dict[tuple[object, ...], list[int]] = {}
        for index, representation in enumerate(representations):
            buckets.setdefault(_representation_key(representation), []).append(
                index
            )

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

        return cls._from_packed_representations(
            packed_representations=tuple(packed_representations),
            member_locations=tuple(
                input_locations[index] for index in member_representation_ids
            ),
        )

    @property
    def num_members(self) -> int:
        """Return the number of logical ensemble members."""
        return len(self._member_locations)

    @property
    def device(self) -> torch.device:
        """Return the common device of all representations."""
        return self._packed_representations[0].device

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

    def _select_members(self, member_ids: Sequence[int]) -> Self:
        locations: dict[tuple[int, int], int] = {}
        representations: list[TableTensor] = []
        member_representation_ids: list[int] = []
        for member_id in member_ids:
            location = self._member_locations[member_id]
            if location not in locations:
                locations[location] = len(representations)
                representations.append(self.representation(member_id))
            member_representation_ids.append(locations[location])
        return self.pack(representations, member_representation_ids)

    def _replace_packed_representations(
        self,
        packed_representations: tuple[TableTensor, ...],
    ) -> Self:
        if tuple(
            representation.size(0) for representation in packed_representations
        ) != tuple(
            representation.size(0)
            for representation in self._packed_representations
        ):
            raise ValueError(
                "Expected replacement representations to preserve packed "
                "sizes."
            )
        return self._from_packed_representations(
            packed_representations=packed_representations,
            member_locations=self._member_locations,
        )

    def materialize(
        self,
        member_ids: Sequence[int] | None = None,
    ) -> TableTensor:
        r"""Materialize members with identical table metadata.

        Args:
            member_ids: Optional member positions. All members are used by
                default.
        """
        if member_ids is None:
            if len(
                self._packed_representations
            ) == 1 and self._member_locations == tuple(
                (0, member) for member in range(self.num_members)
            ):
                return self._packed_representations[0]
            member_ids = range(self.num_members)
        representations = tuple(
            self.representation(member_id) for member_id in member_ids
        )
        reference = _representation_key(representations[0])
        if any(
            _representation_key(representation) != reference
            for representation in representations[1:]
        ):
            raise ValueError(
                "Cannot materialize ensemble members with different metadata."
            )
        return cast(TableTensor, torch.stack(representations, dim=0))
