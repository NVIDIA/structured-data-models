from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Clamp(Processor):
    """Clamp tensor values to fixed lower and upper bounds.

    Unlike :class:`~sdm.processing.Clip`, the bounds are constructor values and
    are not learned from data. The processor is stateless and preserves input
    shape, dtype, and device.

    Args:
        min_value: Optional inclusive lower bound.
        max_value: Optional inclusive upper bound.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        super().__init__()
        if min_value is None and max_value is None:
            raise ValueError("at least one clamp bound must be provided")
        if (
            min_value is not None
            and max_value is not None
            and min_value > max_value
        ):
            raise ValueError("min_value must not exceed max_value")
        self.min_value = min_value
        self.max_value = max_value

    def _transform(self, input: TableTensor) -> TableTensor:
        """Clamp ``input`` to the configured bounds.

        Args:
            input: Numerical table to clamp.

        Returns:
            Table with clamped numerical values and the input schema, dtype,
            and device.
        """
        numerical = input.numerical.clamp(
            min=self.min_value,
            max=self.max_value,
        )
        return input.replace_blocks(numerical=numerical)
