from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch

from sdm import Stype, StypeLike, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class SelectColumns(EnsembleProcessor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep per semantic type.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block. ``"round_robin"`` assigns each ensemble
            member the next consecutive chunk, wrapping to the first column
            after the last. For five columns and ``max_columns=2``, members
            0, 1, and 2 receive columns (0, 1), (2, 3), and (4, 0).
    """

    handles_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(
        self,
        max_columns: int,
        method: Literal["first", "round_robin"] = "first",
    ) -> None:
        super().__init__()
        if max_columns <= 0:
            raise ValueError("max_columns must be positive")
        self.max_columns = max_columns
        self.method = method

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.method == "first":
            groups = []
            for group in ensemble_table:
                columns: Mapping[StypeLike, Sequence[str]] = {
                    stype: column_names[: self.max_columns]
                    for stype, column_names in group.columns.items()
                }
                blocks = {
                    stype: block[..., : self.max_columns]
                    for stype, block in group.items()
                }
                groups.append(group.__class__(columns=columns, **blocks))
            return ensemble_table.replace_groups(groups)

        assert self.method == "round_robin"
        tables: list[TableTensor] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            columns: dict[StypeLike, tuple[str, ...]] = {}
            blocks = {}
            for stype, block in table.items():
                column_names = table.columns[stype]
                num_columns = len(column_names)
                # Empty blocks are passed through unchanged.
                if num_columns == 0:
                    columns[stype] = column_names
                    blocks[stype] = block
                    continue
                chunk_size = min(self.max_columns, num_columns)
                start = member_id * chunk_size % num_columns
                first_count = min(chunk_size, num_columns - start)
                wrap_count = chunk_size - first_count
                # Consecutive chunk from ``start``, wrapping past the last
                # column back to the first.
                columns[stype] = (
                    column_names[start : start + first_count]
                    + column_names[:wrap_count]
                )
                if wrap_count == 0:
                    blocks[stype] = block[..., start : start + first_count]
                    continue
                blocks[stype] = torch.cat(
                    [
                        block[..., start : start + first_count],
                        block[..., :wrap_count],
                    ],
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


class SelectRows(EnsembleProcessor):
    r"""Select a subset of rows.

    Args:
        max_rows: The maximum number of rows to keep.
        method: The row selection method. ``"first"`` keeps the first rows.
            ``"round_robin"`` assigns each ensemble member the next
            consecutive chunk, wrapping to the first row after the last. For
            five rows and ``max_rows=2``, members 0, 1, and 2 receive rows
            (0, 1), (2, 3), and (4, 0).
    """

    handles_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(
        self,
        max_rows: int,
        method: Literal["first", "round_robin"] = "first",
    ) -> None:
        super().__init__()
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.max_rows = max_rows
        self.method = method

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.method == "first":
            return ensemble_table.replace_groups(
                [group[..., : self.max_rows, :] for group in ensemble_table]
            )

        assert self.method == "round_robin"
        tables: list[TableTensor] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            num_rows = table.size(-2)
            if num_rows <= self.max_rows:
                tables.append(table)
                continue

            start = member_id * self.max_rows % num_rows
            length = min(self.max_rows, num_rows - start)
            first = table.narrow(dim=-2, start=start, length=length)

            wrap_length = self.max_rows - length
            if wrap_length == 0:
                tables.append(cast(TableTensor, first))
                continue

            second = table.narrow(dim=-2, start=0, length=wrap_length)
            tables.append(cast(TableTensor, torch.cat([first, second], -2)))

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )
