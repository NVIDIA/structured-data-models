from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from sdm.relational import RelatedTables, Relationship, TaskLink
from sdm.stype import Stype
from sdm.tensor import (
    CategoricalTensor,
    ColumnarTensor,
    EnsembleTable,
    StringTensor,
    TableTensor,
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
class EnsembleRelatedTables:
    r"""Store ensemble representations for every logical related table.

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
                name: table.representation(member_id)
                for name, table in self.tables.items()
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
