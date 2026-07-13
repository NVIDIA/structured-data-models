from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class HardClip(Processor):
    """Clamp numerical values to fixed lower and upper bounds.

    Unlike :class:`~sdm.processing.Clip`, the bounds are configuration rather
    than fitted quantiles. The transformation is not invertible because values
    outside the interval are discarded.

    Args:
        min_value: Inclusive lower bound.
        max_value: Inclusive upper bound.
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

    def _transform(self, input: TableTensor) -> TableTensor:
        """Clamp ``input`` to the configured interval."""
        numerical = _as_float(input.numerical).clamp(
            min=self.min_value,
            max=self.max_value,
        )
        return input.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"min_value={self.min_value}, max_value={self.max_value})"
        )
