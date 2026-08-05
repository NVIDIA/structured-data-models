import torch

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Softmax(Processor):
    """Apply softmax to logits after temperature scaling.

    Softmax acts on the final class dimension and preserves all leading
    dimensions, so it supports both stacked and reduced estimator outputs.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    supported_stypes = frozenset({Stype.numerical})
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
        numerical = torch.softmax(
            table.numerical / self.temperature,
            dim=-1,
        )
        return table.replace_blocks(numerical=numerical)
