from sdm import Stype, TableTensor
from sdm.processing import Processor


class Clip(Processor):
    """Clamp numerical values to a fixed interval.

    Values below ``min_value`` are set to ``min_value``, and values above
    ``max_value`` are set to ``max_value``. Values within the interval are
    unchanged. This processor is stateless, so it can transform a table
    without being fitted first.

    Args:
        min_value: Inclusive lower bound for every numerical value.
        max_value: Inclusive upper bound for every numerical value.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.clamp(self.min_value, self.max_value)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"{self.min_value}, {self.max_value})"
        )
