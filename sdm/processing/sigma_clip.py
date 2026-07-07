import torch
from torch import Tensor

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


def _nanstd(input: Tensor, *, dim: int) -> Tensor:
    mask = ~torch.isnan(input)
    count = mask.sum(dim=dim)
    mean = torch.nanmean(input, dim=dim)
    centered = input - mean
    centered = torch.where(mask, centered, torch.zeros_like(centered))
    sum_squares = centered.square().sum(dim=dim)

    correction = (count > 1).to(count.dtype)
    denominator = (count - correction).clamp_min(1)
    variance = sum_squares / denominator
    return torch.where(count > 0, variance.sqrt(), torch.nan)


class SigmaClip(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    The first pass masks values outside the initial z-score bounds, then the
    second pass refits bounds on the remaining values. The transform applies
    logarithmic soft clipping instead of hard truncation.

    Args:
        threshold: Positive z-score multiplier setting how many standard
            deviations from the mean mark the soft clipping bounds.
    """

    def __init__(
        self,
        *,
        threshold: float = 4.0,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self.register_buffer("_mean", torch.empty(0))
        self.register_buffer("_std", torch.empty(0))
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(self, input: TableTensor) -> None:
        numerical = _as_float(input.numerical)
        min_std = numerical.new_tensor(1e-6)

        mean = torch.nanmean(numerical, dim=0)
        std = _nanstd(numerical, dim=0)
        std = torch.where(std.isnan(), min_std, std)
        std = torch.maximum(std, min_std)

        inf = numerical.new_tensor(float("inf"))
        lower_bound = torch.where(
            mean.isnan(), -inf, mean - self.threshold * std
        )
        upper_bound = torch.where(
            mean.isnan(), inf, mean + self.threshold * std
        )
        outlier_mask = (numerical < lower_bound) | (numerical > upper_bound)
        clean = torch.where(outlier_mask, torch.nan, numerical)

        mean_clean = torch.nanmean(clean, dim=0)
        std_clean = _nanstd(clean, dim=0)
        self._mean = torch.where(mean_clean.isnan(), mean, mean_clean)
        self._std = torch.where(std_clean.isnan(), std, std_clean)
        self._std = torch.maximum(self._std, min_std)
        self.lower_bound = torch.where(
            self._mean.isnan(),
            -inf,
            self._mean - self.threshold * self._std,
        )
        self.upper_bound = torch.where(
            self._mean.isnan(),
            inf,
            self._mean + self.threshold * self._std,
        )

    def forward(self, input: TableTensor) -> TableTensor:
        """Clip ``input`` using the fitted soft lower and upper bounds."""
        numerical = _as_float(input.numerical)
        log_abs = numerical.abs().log1p()
        clipped = torch.maximum(-log_abs + self.lower_bound, numerical)
        numerical = torch.minimum(log_abs + self.upper_bound, clipped)
        return input.replace_blocks(numerical=numerical)
