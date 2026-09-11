import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ImputeMean(Processor):
    """Replace NaN feature values with fitted per-column means.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (e.g. all-NaN columns).
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        # Match nanmean without widening a full-table mask to int64.
        count_dtype = (
            torch.int32 if numerical.size(-2) <= 2**31 - 1 else torch.int64
        )
        count = numerical.isnan().sum(
            dim=-2,
            keepdim=True,
            dtype=count_dtype,
        )
        count.neg_().add_(numerical.size(-2))
        mean = numerical.nansum(dim=-2, keepdim=True).div_(count)
        self._mean = mean.masked_fill_(mean.isnan(), self.fill_value)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace NaNs with the fitted per-column means."""
        numerical = table.numerical
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        return table.replace_blocks(numerical=numerical)
