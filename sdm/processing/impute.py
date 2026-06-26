import torch
from torch import Tensor

from sdm.processing._utils import _ensure_floating
from sdm.processing.base import Processor


class MeanImpute(Processor):
    """Replace missing feature values with fitted per-column means.

    Args:
        fill_value: Value used for columns whose fitted mean is undefined
            (e.g. all-NaN columns).
    """

    def __init__(self, *, fill_value: float = 0.0) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("mean", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        input = _ensure_floating(input)
        mean = torch.nanmean(input, dim=0)
        empty = mean.new_full(mean.shape, self.fill_value)
        self.mean = torch.where(torch.isnan(mean), empty, mean)

    def forward(self, input: Tensor) -> Tensor:
        """Replace NaNs with the fitted per-column means."""
        input = _ensure_floating(input)
        return torch.where(
            torch.isnan(input), self.mean.expand_as(input), input
        )
