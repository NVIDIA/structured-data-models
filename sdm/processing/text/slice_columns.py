from typing import Literal

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class SliceColumns(Processor):
    """Keep up to ``max_columns`` numerical columns.

    A stateless column reduction, for keeping the leading columns of text
    embeddings. Tables with at most ``max_columns`` numerical columns pass
    through unchanged.

    Args:
        max_columns: Maximum number of numerical columns to keep.
        mode: Column selection mode. Currently only ``"first"`` is supported.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        max_columns: int,
        mode: Literal["first"] = "first",
    ) -> None:
        super().__init__()
        if max_columns < 1:
            raise ValueError(
                f"'max_columns' must be positive (got {max_columns})"
            )
        if mode != "first":
            raise ValueError(f"Unsupported mode (got {mode!r})")

        self.max_columns = max_columns

    def _transform(self, table: TableTensor) -> TableTensor:
        names = table.columns[Stype.numerical]
        if len(names) <= self.max_columns:
            return table
        return table.select_columns(names[: self.max_columns])
