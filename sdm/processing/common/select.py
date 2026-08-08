from collections.abc import Mapping, Sequence
from typing import Literal

from sdm import Stype, StypeLike, TableTensor
from sdm.processing import Processor


class SelectColumns(Processor):
    r"""Select a subset of columns for each semantic type.

    Args:
        max_columns: The maximum number of columns to keep.
        method: The column selection method.
            ``"first"`` keeps the first columns according to their order within
            each semantic block.
    """

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(
        self,
        max_columns: int,
        method: Literal["first"] = "first",
    ) -> None:
        super().__init__()
        self.max_columns = max_columns
        self.method = method

    def _transform(self, table: TableTensor) -> TableTensor:
        if self.method == "first":
            columns: Mapping[StypeLike, Sequence[str]] = {
                stype: columns[: self.max_columns]
                for stype, columns in table.columns.items()
            }
            blocks = {
                stype: block[..., : self.max_columns]
                for stype, block in table.items()
            }
            return table.__class__(columns=columns, **blocks)

        raise NotImplementedError
