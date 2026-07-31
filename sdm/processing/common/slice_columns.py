from collections.abc import Mapping
from typing import Literal

from sdm.processing.base import Processor
from sdm.stype import Stype, StypeLike
from sdm.tensor import TableTensor


class SliceColumns(Processor):
    """Limit the number of columns for each semantic type.

    Args:
        max_columns: Maximum number of columns to keep. An integer applies to
            every semantic type. A mapping applies limits by semantic type;
            unconfigured types pass through unchanged.
        mode: ``"first"`` keeps the leading columns of each configured
            semantic type.
    """

    requires_fit = False
    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *,
        max_columns: int | Mapping[StypeLike, int],
        mode: Literal["first"] = "first",
    ) -> None:
        super().__init__()
        if isinstance(max_columns, int):
            self.max_columns = dict.fromkeys(Stype, max_columns)
        else:
            self.max_columns = {
                Stype(stype): maximum for stype, maximum in max_columns.items()
            }

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = tuple(
            name
            for stype, names in table.columns.items()
            for name in names[: self.max_columns.get(stype, len(names))]
        )
        if len(columns) == table.size(-1):
            return table
        return table.select_columns(columns)
