import torch

from sdm import NaT, Stype, TableTensor
from sdm.processing import Processor


class ImputeMean(Processor):
    """Replace missing numerical and datetime values with fitted means.

    Args:
        fill_value: Finite value used for numerical columns whose fitted mean
            is undefined. All-missing datetime columns remain missing.
    """

    handles_stypes = frozenset({Stype.numerical, Stype.datetime})
    requires_fit = True

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.register_buffer("_mean", torch.empty(0))
        self.register_buffer(
            "_datetime_mean", torch.empty(0, dtype=torch.int64)
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        mean = torch.nanmean(
            numerical,
            dim=-2,
            keepdim=True,
        )
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

        datetime = table.datetime
        datetime_float = datetime.to(torch.float64)
        datetime_float = datetime_float.masked_fill(
            datetime == NaT, float("nan")
        )
        self._datetime_mean = (
            datetime_float.nanmean(dim=-2, keepdim=True)
            .round()
            .nan_to_num(nan=NaT)
            .to(torch.int64)
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace missing values with the fitted per-column means."""
        numerical = table.numerical
        numerical = torch.where(numerical.isnan(), self._mean, numerical)
        datetime = table.datetime
        datetime = torch.where(
            datetime == NaT,
            self._datetime_mean,
            datetime,
        )
        return table.replace_blocks(numerical=numerical, datetime=datetime)
