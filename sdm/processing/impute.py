import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class MeanImpute(Processor):
    """Replace missing feature values with fitted per-column means.

    Args:
        fill_value: Value used for columns whose fitted mean is undefined
            (e.g. all-NaN columns).
    """

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))

    def _fit(self, input: TableTensor) -> None:
        numerical = _as_float(input.numerical)
        mean = torch.nanmean(numerical, dim=0)
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

    def forward(self, input: TableTensor) -> TableTensor:
        """Replace NaNs with the fitted per-column means."""
        self._check_supported_stypes(input)
        numerical = _as_float(input.numerical)
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        return input.replace_blocks(numerical=numerical)
