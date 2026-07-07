import torch
from torch import Tensor

from sdm.processing._utils import _as_float
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
        self.register_buffer("_mean", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        input = _as_float(input)
        mean = torch.nanmean(input, dim=0)
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

    def _transform(self, input: Tensor) -> Tensor:
        """Replace NaNs with the fitted per-column means."""
        input = _as_float(input)
        return torch.where(input.isnan(), self._mean, input)
