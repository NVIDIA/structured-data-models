import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class Softmax(Processor):
    """Apply softmax to logits after temperature scaling.

    Softmax acts on the final class dimension and preserves all leading
    dimensions, so it supports both stacked and reduced estimator outputs.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``softmax(table / temperature)`` over the last dimension."""
        if self.temperature == 1.0 and table.numerical.is_floating_point():
            numerical = table.numerical.softmax(dim=-1)
        else:
            numerical = table.numerical / self.temperature
            if torch.is_autocast_enabled(numerical.device.type):
                numerical = numerical.softmax(dim=-1)
            else:
                torch.softmax(numerical, dim=-1, out=numerical)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        if self.temperature == 1.0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"temperature={self.temperature})"
        )
