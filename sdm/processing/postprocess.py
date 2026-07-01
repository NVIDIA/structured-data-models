"""Postprocessing transforms for model outputs."""

import math

import torch
from torch import Tensor

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor


class SoftmaxTemperature(Processor):
    """Apply softmax to logits after temperature scaling.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    requires_fit = False

    def __init__(self, *, temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.temperature = temperature

    def forward(self, input: Tensor) -> Tensor:
        """Return ``softmax(input / temperature)`` over the last dimension.

        Args:
            input: Logit tensor with shape ``[..., C]``.

        Returns:
            Probability tensor with shape ``[..., C]``.
        """
        return torch.softmax(
            _as_float(input) / self.temperature,
            dim=-1,
        )
