from collections.abc import Mapping, Sequence
from typing import Literal

import torch

from sdm import Stype, StypeLike
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class SelectColumns(EnsembleProcessor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep per ensemble member.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block. ``"round_robin"`` assigns consecutive column
            chunks to ensemble members.
    """

    handles_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(
        self,
        max_columns: int,
        method: Literal["first", "round_robin"] = "first",
    ) -> None:
        super().__init__()
        self.max_columns = max_columns
        self.method = method

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.method == "first":
            new_groups = []
            for group in ensemble_table:
                new_columns: Mapping[StypeLike, Sequence[str]] = {
                    stype: column_names[: self.max_columns]
                    for stype, column_names in group.columns.items()
                }
                new_blocks = {
                    stype: block[..., : self.max_columns]
                    for stype, block in group.items()
                }
                new_groups.append(
                    group.__class__(columns=new_columns, **new_blocks)
                )
            return ensemble_table.replace_groups(new_groups)

        assert self.method == "round_robin"
        tables = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            new_columns: dict[StypeLike, tuple[str, ...]] = {}
            new_blocks = {}
            for stype, block in table.items():
                column_names = table.columns[stype]
                column_count = min(self.max_columns, len(column_names))
                start_index = (
                    0
                    if column_count == len(column_names)
                    else member_id * self.max_columns
                )
                indices = (
                    tuple(
                        (start_index + offset) % len(column_names)
                        for offset in range(column_count)
                    )
                    if column_count > 0
                    else ()
                )
                new_columns[stype] = tuple(
                    column_names[index] for index in indices
                )
                if indices == tuple(range(len(column_names))):
                    new_blocks[stype] = block
                    continue
                if len(indices) == 0:
                    new_blocks[stype] = block[..., :0]
                    continue
                new_blocks[stype] = torch.cat(
                    [block.narrow(-1, index, 1) for index in indices],
                    dim=-1,
                )
            tables.append(
                table.__class__(
                    columns=new_columns,
                    **new_blocks,
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )
