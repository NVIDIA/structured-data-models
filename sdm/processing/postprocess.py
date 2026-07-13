import math

import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class SoftmaxTemperature(Processor):
    """Apply softmax to logits after temperature scaling.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    requires_fit = False

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.temperature = temperature

    def _transform(self, inp: TableTensor) -> TableTensor:
        """Return ``softmax(input / temperature)`` over the last dimension."""
        numerical = torch.softmax(
            _as_float(inp.numerical) / self.temperature,
            dim=-1,
        )
        return inp.replace_blocks(numerical=numerical)
