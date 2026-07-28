import math

import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class ImputeMean(Processor):
    """Replace NaN feature values with fitted per-column means.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (e.g. all-NaN columns).
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(fill_value):
            raise ValueError("fill_value must be finite.")
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = _as_float(table.numerical)
        mean = torch.nanmean(numerical, dim=0)
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace NaNs with the fitted per-column means."""
        numerical = _as_float(table.numerical)
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        return table.replace_blocks(numerical=numerical)
