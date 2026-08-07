import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


def _std(
    inp: Tensor,
    *,
    dim: int,
) -> Tensor:
    correction = 1 if inp.size(dim) > 1 else 0
    return inp.std(dim=dim, correction=correction, keepdim=True)


class ClipSigma(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    The first pass masks values outside the initial z-score bounds, then the
    second pass refits bounds on the remaining values. The transform applies
    logarithmic soft clipping instead of hard truncation.

    Args:
        threshold: Positive z-score multiplier setting how many standard
            deviations from the mean mark the soft clipping bounds.
    """

    operates_on_stypes = frozenset({Stype.numerical})

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

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        min_std = numerical.new_tensor(1e-6)

        mean = numerical.mean(dim=-2, keepdim=True)
        std = torch.maximum(
            _std(
                numerical,
                dim=-2,
            ),
            min_std,
        )
        lower_bound = mean - self.threshold * std
        upper_bound = mean + self.threshold * std
        outlier_mask = (numerical < lower_bound) | (numerical > upper_bound)

        keep = ~outlier_mask
        count = keep.sum(dim=-2, keepdim=True)
        safe_count = count.clamp_min(1)
        clean_sum = torch.where(
            keep,
            numerical,
            0.0,
        ).sum(dim=-2, keepdim=True)
        mean_clean = clean_sum / safe_count
        centered = torch.where(
            keep,
            numerical - mean_clean,
            0.0,
        )
        correction = (count > 1).to(count.dtype)
        denominator = (count - correction).clamp_min(1)
        std_clean = (
            centered.square().sum(dim=-2, keepdim=True) / denominator
        ).sqrt()

        has_clean = count > 0
        self._mean = torch.where(has_clean, mean_clean, mean)
        self._std = torch.where(has_clean, std_clean, std)
        self._std = torch.maximum(self._std, min_std)
        self.lower_bound = self._mean - self.threshold * self._std
        self.upper_bound = self._mean + self.threshold * self._std

    def _transform(self, table: TableTensor) -> TableTensor:
        """Clip ``table`` using the fitted soft lower and upper bounds."""
        numerical = table.numerical
        log_abs = numerical.abs().log1p()
        clipped = torch.maximum(-log_abs + self.lower_bound, numerical)
        numerical = torch.minimum(log_abs + self.upper_bound, clipped)
        return table.replace_blocks(numerical=numerical)
