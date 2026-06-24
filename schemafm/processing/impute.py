from __future__ import annotations

import torch
from torch import Tensor

from schemafm.processing.base import Processor


class MeanImpute(Processor):
    """Replace missing feature values with fitted per-column means."""

    def __init__(self, *, empty_value: float = 0.0) -> None:
        super().__init__()
        self.empty_value = empty_value
        self.register_buffer("mean", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        mean = torch.nanmean(input, dim=0)
        empty = mean.new_full(mean.shape, self.empty_value)
        self.mean = torch.where(torch.isnan(mean), empty, mean)

    def forward(self, input: Tensor) -> Tensor:
        """Replace NaNs with the fitted per-column means."""
        self._check_is_fitted()
        return torch.where(
            torch.isnan(input), self.mean.expand_as(input), input
        )
