from __future__ import annotations

import torch
from torch import Tensor

from schemafm.processing.base import Processor


class SoftmaxTemperature(Processor):
    """Apply softmax to logits after temperature scaling."""

    requires_fit = False

    def __init__(self, *, temperature: float = 1.0, dim: int = -1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Return ``softmax(input / temperature)`` along ``dim``."""
        return torch.softmax(input / self.temperature, dim=self.dim)
