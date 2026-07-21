from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class SliceFeatures(Processor):
    """Keep only the first ``dim`` numerical columns.

    A stateless dimension reduction, e.g. for slicing the leading dimensions
    of text embeddings. Tables with at most ``dim`` columns pass through
    unchanged.

    Args:
        dim: Number of leading columns to keep.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"'dim' must be positive (got {dim})")
        self.dim = dim

    def _transform(self, table: TableTensor) -> TableTensor:
        names = table.columns[Stype.numerical]
        if len(names) <= self.dim:
            return table
        return table.select_columns(names[: self.dim])
