from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from typing_extensions import Self

from sdm.relational import RelatedTables, Relationship, TaskLink
from sdm.stype import Stype
from sdm.tensor import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    TableTensor,
)


def _variant_metadata_key(table: TableTensor) -> tuple[object, ...]:
    categorical = tuple(
        (
            id(category),
            tuple(category.size()),
            category.dtype,
            category.device,
        )
        for category in table.categorical.categories
    )
    return (
        tuple(table.size()),
        tuple((stype, columns) for stype, columns in table.columns.items()),
        tuple(
            (stype, type(block), block.dtype) for stype, block in table.items()
        ),
        table.device,
        categorical,
    )


def _stack_positional(tables: Sequence[TableTensor]) -> TableTensor:
    if len(tables) == 0:
        raise ValueError("Expected at least one table to materialize.")
    if len(tables) == 1:
        return cast(TableTensor, tables[0].unsqueeze(0))

    reference = tables[0]
    reference_size = tuple(reference.size()[:-1])
    reference_widths = {
        stype: block.size(-1) for stype, block in reference.items()
    }
    for table in tables[1:]:
        if tuple(table.size()[:-1]) != reference_size:
            raise ValueError(
                "Cannot materialize ensemble members with different row or "
                "batch dimensions."
            )
        widths = {stype: block.size(-1) for stype, block in table.items()}
        if widths != reference_widths:
            raise ValueError(
                "Cannot materialize ensemble members with incompatible "
                "semantic-type widths."
            )
        for (stype, actual), (_, expected) in zip(
            table.items(),
            reference.items(),
        ):
            if actual.dtype != expected.dtype:
                raise ValueError(
                    "Cannot materialize ensemble members with different "
                    f"{stype.value!r} dtypes."
                )
            if actual.device != expected.device:
                raise ValueError(
                    "Cannot materialize ensemble members on different devices."
                )

    blocks: dict[Stype, torch.Tensor] = {}
    for stype, reference_block in reference.items():
        member_blocks = [table.blocks[stype] for table in tables]
        if stype == Stype.categorical:
            categories = reference.categorical.categories
            expected_counts = tuple(
                category.numel() for category in categories
            )
            for table in tables[1:]:
                if (
                    tuple(
                        category.numel()
                        for category in table.categorical.categories
                    )
                    != expected_counts
                ):
                    raise ValueError(
                        "Cannot materialize categorical ensemble members "
                        "with different class counts."
                    )
            code = torch.stack(
                [table.categorical.code for table in tables],
                dim=0,
            )
            blocks[stype] = CategoricalTensor(
                code=code,
                categories=categories,
            )
        elif reference_block.size(-1) > 0:
            blocks[stype] = torch.stack(member_blocks, dim=0)

    return TableTensor(
        size=(len(tables), *reference_size),
        columns={
            stype.value: columns
            for stype, columns in reference.columns.items()
        },
        device=reference.device,
        numerical=blocks.get(Stype.numerical),
        categorical=cast(
            CategoricalTensor | None,
            blocks.get(Stype.categorical),
        ),
        datetime=blocks.get(Stype.datetime),
        text=cast(StringTensor | None, blocks.get(Stype.text)),
        id=cast(ColumnarTensor | None, blocks.get(Stype.id)),
    )


@dataclass(frozen=True)
class EnsembleTable:
    r"""Store shared and distinct table variants for ensemble members.

    Every group has shape ``[V, ..., R, C]``. The stable member mapping points
    to a group and variant position without inferring equality from tensor
    values.

    Args:
        groups: Compatible table variants grouped along their leading
            dimension.
        member_to_variant: ``(group, variant)`` location for every member.
    """

    groups: tuple[TableTensor, ...]
    member_to_variant: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.groups) == 0:
            raise ValueError("Expected at least one ensemble group.")
        if len(self.member_to_variant) == 0:
            raise ValueError("Expected at least one ensemble member.")

        devices = {group.device for group in self.groups}
        if len(devices) != 1:
            raise ValueError(
                "Expected all ensemble groups to use the same device."
            )

        for group_index, group in enumerate(self.groups):
            if group.dim() < 3:
                raise ValueError(
                    "Expected ensemble groups with shape "
                    f"[V, ..., R, C] (group {group_index} is {group.dim()}D)."
                )
            if group.size(0) == 0:
                raise ValueError(
                    "Expected every ensemble group to be non-empty."
                )

        for member, (group, variant) in enumerate(self.member_to_variant):
            if not 0 <= group < len(self.groups):
                raise ValueError(
                    f"Member {member} references unknown group {group}."
                )
            if not 0 <= variant < self.groups[group].size(0):
                raise ValueError(
                    f"Member {member} references unknown variant {variant} "
                    f"in group {group}."
                )

    @classmethod
    def from_shared(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> Self:
        r"""Create an ensemble whose members share one table.

        Args:
            table: Shared table with shape ``[..., R, C]``.
            num_members: Positive number of logical ensemble members.
        """
        if num_members < 1:
            raise ValueError("'num_members' needs to be positive.")
        group = cast(TableTensor, table.unsqueeze(0))
        return cls(
            groups=(group,),
            member_to_variant=((0, 0),) * num_members,
        )

    @classmethod
    def pack(
        cls,
        variants: Sequence[TableTensor],
        member_to_input_variant: Sequence[int],
    ) -> Self:
        r"""Pack proven variants into compatible physical groups.

        Args:
            variants: Distinct results in provenance order.
            member_to_input_variant: Input variant index for every member.
        """
        variants = tuple(variants)
        member_to_input_variant = tuple(member_to_input_variant)
        if len(variants) == 0:
            raise ValueError("Expected at least one input variant.")
        if len(member_to_input_variant) == 0:
            raise ValueError("Expected at least one ensemble member.")
        if any(
            variant < 0 or variant >= len(variants)
            for variant in member_to_input_variant
        ):
            raise ValueError(
                "'member_to_input_variant' references an unknown variant."
            )

        buckets: dict[tuple[object, ...], list[int]] = {}
        for index, table in enumerate(variants):
            buckets.setdefault(_variant_metadata_key(table), []).append(index)

        groups: list[TableTensor] = []
        input_to_location: dict[int, tuple[int, int]] = {}
        for indices in buckets.values():
            group_index = len(groups)
            group = (
                cast(TableTensor, variants[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        [variants[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            groups.append(group)
            input_to_location.update(
                {
                    input_index: (group_index, variant_index)
                    for variant_index, input_index in enumerate(indices)
                }
            )

        return cls(
            groups=tuple(groups),
            member_to_variant=tuple(
                input_to_location[index] for index in member_to_input_variant
            ),
        )

    @property
    def num_members(self) -> int:
        """Return the number of logical ensemble members."""
        return len(self.member_to_variant)

    @property
    def device(self) -> torch.device:
        """Return the common device of all groups."""
        return self.groups[0].device

    def __getitem__(self, member_id: int) -> TableTensor:
        group, variant = self.member_to_variant[member_id]
        return self.groups[group][variant]

    def _select_members(self, member_ids: Sequence[int]) -> Self:
        r"""Select members while preserving their requested order.

        Args:
            member_ids: Member positions to select.
        """
        member_ids = tuple(member_ids)
        locations: dict[tuple[int, int], int] = {}
        variants: list[TableTensor] = []
        member_to_input: list[int] = []
        for member_id in member_ids:
            location = self.member_to_variant[member_id]
            if location not in locations:
                locations[location] = len(variants)
                group, variant = location
                variants.append(self.groups[group][variant])
            member_to_input.append(locations[location])
        return self.pack(
            variants=variants,
            member_to_input_variant=member_to_input,
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
            if len(self.groups) == 1 and self.member_to_variant == tuple(
                (0, member) for member in range(self.num_members)
            ):
                return self.groups[0]
            member_ids = tuple(range(self.num_members))
        tables = tuple(self[member_id] for member_id in member_ids)
        reference = _variant_metadata_key(tables[0])
        if any(
            _variant_metadata_key(table) != reference for table in tables[1:]
        ):
            raise ValueError(
                "Cannot materialize ensemble members with different metadata."
            )
        return cast(TableTensor, torch.stack(tables, dim=0))

    def with_groups(self, groups: tuple[TableTensor, ...]) -> Self:
        r"""Replace physical groups while retaining the member mapping.

        Args:
            groups: Replacement groups with unchanged variant counts.
        """
        if len(groups) != len(self.groups) or any(
            actual.size(0) != expected.size(0)
            for actual, expected in zip(groups, self.groups)
        ):
            raise ValueError(
                "Expected replacement groups to preserve group and variant "
                "counts."
            )
        return self.__class__(
            groups=groups,
            member_to_variant=self.member_to_variant,
        )


@dataclass(frozen=True)
class EnsembleRelatedTables:
    r"""Store ensemble variants for every logical related table.

    Args:
        tables: Ensemble tables keyed by logical table name.
        relationships: Shared relationships among the tables.
        task_links: Shared links from task rows to related tables.
    """

    tables: Mapping[str, EnsembleTable]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def __post_init__(self) -> None:
        member_counts = {table.num_members for table in self.tables.values()}
        if len(member_counts) > 1:
            raise ValueError(
                "Expected every related ensemble table to have the same "
                "number of members."
            )
        devices = {table.device for table in self.tables.values()}
        if len(devices) > 1:
            raise ValueError(
                "Expected every related ensemble table on the same device."
            )

    def __getitem__(self, table_name: str) -> EnsembleTable:
        return self.tables[table_name]

    def member(self, member_id: int) -> RelatedTables:
        r"""Return one member's related tables.

        Args:
            member_id: Stable member position.
        """
        return RelatedTables(
            tables={
                name: table[member_id] for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )

    def materialize(self, member_ids: Sequence[int]) -> RelatedTables:
        r"""Materialize selected members for model execution.

        Args:
            member_ids: Stable member positions.
        """
        return RelatedTables(
            tables={
                name: table.materialize(member_ids)
                for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )
