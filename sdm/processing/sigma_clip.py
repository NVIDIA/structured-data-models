import torch
from torch import Tensor

from sdm.processing.base import Processor


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
    return torch.where(
        count > 0, variance.sqrt(), torch.full_like(variance, torch.nan)
    )


class SigmaClip(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    Args:
        threshold: Positive z-score multiplier setting how many standard
            deviations from the mean mark the soft clipping bounds.
    """

    def __init__(self, *, threshold: float = 4.0) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("std", torch.empty(0))
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        min_std = input.new_tensor(1e-6)

        mean = torch.nanmean(input, dim=0)
        std = torch.maximum(
            _nanstd(input, dim=0),
            min_std,
        )

        lower_bound = mean - self.threshold * std
        upper_bound = mean + self.threshold * std
        outlier_mask = (input < lower_bound) | (input > upper_bound)
        clean = torch.where(
            outlier_mask, input.new_full(input.shape, torch.nan), input
        )

        self.mean = torch.nanmean(clean, dim=0)
        self.std = torch.maximum(
            _nanstd(clean, dim=0),
            min_std,
        )
        self.lower_bound = self.mean - self.threshold * self.std
        self.upper_bound = self.mean + self.threshold * self.std

    def forward(self, input: Tensor) -> Tensor:
        """Clip ``input`` using the fitted soft lower and upper bounds."""
        log_abs = torch.log1p(input.abs())
        clipped = torch.maximum(-log_abs + self.lower_bound, input)
        return torch.minimum(log_abs + self.upper_bound, clipped)
