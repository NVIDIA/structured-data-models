import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ImputeMean(Processor):
    """Replace missing feature values with fitted per-column means.

    Args:
        fill_value: Finite value used for columns whose fitted mean is
            undefined (e.g. all-NaN columns).
        nonfinite: If ``True``, treat positive and negative infinity as
            missing in addition to NaN.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        fill_value: float = 0.0,
        nonfinite: bool = False,
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.nonfinite = nonfinite
        self.register_buffer("_mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        if self.nonfinite:
            numerical = numerical.masked_fill(
                ~numerical.isfinite(),
                float("nan"),
            )
        mean = torch.nanmean(
            numerical,
            dim=-2,
            keepdim=True,
        )
        self._mean = torch.where(mean.isnan(), self.fill_value, mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Replace missing values with the fitted per-column means."""
        numerical = table.numerical
        missing = (
            ~numerical.isfinite() if self.nonfinite else numerical.isnan()
        )
        numerical = torch.where(missing, self._mean, numerical)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self.fill_value != 0.0:
            arguments.append(f"fill_value={self.fill_value!r}")
        if self.nonfinite:
            arguments.append("nonfinite=True")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
