from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


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
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        super().__init__()
        if min_value > max_value:
            raise ValueError(
                "min_value must be less than or equal to max_value"
            )
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, inp: TableTensor) -> TableTensor:
        """Clamp ``inp`` to the configured interval."""
        numerical = _as_float(inp.numerical).clamp(
            min=self.min_value,
            max=self.max_value,
        )
        return inp.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"min_value={self.min_value}, max_value={self.max_value})"
        )
