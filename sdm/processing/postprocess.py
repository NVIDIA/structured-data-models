import math

import torch
from torch import Tensor

from sdm.processing.base import Processor


class SoftmaxTemperature(Processor):
    """Apply softmax to logits after temperature scaling.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
        dim: Dimension along which softmax is computed.
    """

    requires_fit = False

    def __init__(self, *, temperature: float = 1.0, dim: int = -1) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.temperature = temperature
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Return ``softmax(input / temperature)`` along ``dim``."""
        return torch.softmax(input / self.temperature, dim=self.dim)
