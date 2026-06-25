from __future__ import annotations

import torch
from torch import Tensor

from schemafm.processing.base import Processor
from schemafm.tensor import CategoricalTensor


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
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        """Return ``softmax(input / temperature)`` along ``dim``."""
        return torch.softmax(input / self.temperature, dim=self.dim)


class DecodeLabels(Processor):
    """Decode class indices using caller-supplied target categories."""

    requires_fit = False

    def forward(
        self, input: Tensor, categories: Tensor | CategoricalTensor
    ) -> Tensor:
        """Return raw labels for ``input`` indices from ``categories``."""
        category_vector = _category_vector(categories)
        if category_vector.dim() != 1:
            raise ValueError("DecodeLabels expects 1D categories.")
        if input.dtype == torch.bool or input.is_floating_point():
            raise ValueError("DecodeLabels expects integer class indices.")

        codes = input.to(torch.long)
        valid = (codes >= 0) & (codes < category_vector.numel())
        if not bool(valid.all()):
            raise ValueError(
                "DecodeLabels received class indices outside categories."
            )
        return category_vector[codes]


def _category_vector(categories: Tensor | CategoricalTensor) -> Tensor:
    if isinstance(categories, CategoricalTensor):
        if len(categories.categories) != 1:
            raise ValueError(
                "DecodeLabels expects a single target category vector."
            )
        return categories.categories[0]
    return categories
