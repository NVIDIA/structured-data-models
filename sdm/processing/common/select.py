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

    supported_stypes = frozenset(Stype)
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
            groups = []
            for table in ensemble_table:
                columns: Mapping[StypeLike, Sequence[str]] = {
                    stype: names[: self.max_columns]
                    for stype, names in table.columns.items()
                }
                blocks = {
                    stype: block[..., : self.max_columns]
                    for stype, block in table.items()
                }
                groups.append(table.__class__(columns=columns, **blocks))
            return ensemble_table.replace_groups(groups)

        assert self.method == "round_robin"
        tables = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            columns: dict[StypeLike, tuple[str, ...]] = {}
            blocks = {}
            for stype, block in table.items():
                names = table.columns[stype]
                count = min(self.max_columns, len(names))
                start = (
                    0 if count == len(names) else member_id * self.max_columns
                )
                indices = (
                    tuple(
                        (start + offset) % len(names)
                        for offset in range(count)
                    )
                    if count > 0
                    else ()
                )
                columns[stype] = tuple(names[index] for index in indices)
                if indices == tuple(range(len(names))):
                    blocks[stype] = block
                    continue
                if len(indices) == 0:
                    blocks[stype] = block[..., :0]
                    continue
                blocks[stype] = torch.cat(
                    [block.narrow(-1, index, 1) for index in indices],
                    dim=-1,
                )
            tables.append(
                table.__class__(
                    columns=columns,
                    **blocks,
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )
