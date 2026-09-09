from sdm import Stype, TableTensor
from sdm.processing import Processor


class ReplaceInf(Processor):
    """Replace infinite numerical values with NaN."""

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        numerical = numerical.masked_fill(numerical.isinf(), float("nan"))
        return table.replace_blocks(numerical=numerical)
