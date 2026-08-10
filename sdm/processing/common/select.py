from typing import Literal

from sdm import Stype, StypeLike, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class SelectColumns(EnsembleProcessor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep per semantic type
            and ensemble member.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block. ``"round_robin"`` assigns consecutive column
            chunks to ensemble members and wraps within each semantic block.
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
            return ensemble_table.replace_groups(
                [self._select(group, member_id=0) for group in ensemble_table]
            )

        assert self.method == "round_robin"
        tables = [
            self._select(
                ensemble_table.table(member_id),
                member_id=member_id,
            )
            for member_id in range(ensemble_table.num_members)
        ]
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _select(
        self,
        table: TableTensor,
        *,
        member_id: int,
    ) -> TableTensor:
        columns: dict[StypeLike, tuple[str, ...]] = {}
        blocks = {}
        for stype, block in table.items():
            names = table.columns[stype]
            if self.max_columns == 0:
                columns[stype] = names[:0]
                blocks[stype] = block[..., :0]
                continue
            num_chunks = max(
                1,
                (len(names) + self.max_columns - 1) // self.max_columns,
            )
            start = member_id % num_chunks * self.max_columns
            stop = start + self.max_columns
            columns[stype] = names[start:stop]
            blocks[stype] = block[..., start:stop]
        return table.__class__(columns=columns, **blocks)
