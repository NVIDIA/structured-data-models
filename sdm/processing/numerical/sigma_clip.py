import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ClipSigma(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    The first pass masks values outside the initial z-score bounds, then the
    second pass refits bounds on the remaining values. The transform applies
    logarithmic soft clipping instead of hard truncation. NaN and infinite
    values are ignored when fitting statistics and preserved during the
    transform.

    Args:
        threshold: Positive z-score multiplier setting how many standard
            deviations from the mean mark the soft clipping bounds.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        threshold: float = 4.0,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        self.threshold = threshold
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        finite = table.numerical.isfinite()
        finite_or_nan = table.numerical.masked_fill(~finite, torch.nan)

        # Compute finite mean and standard deviation:
        mean = finite_or_nan.nanmean(-2, keepdim=True)
        mean.masked_fill_(mean.isnan(), 0.0)

        var = (finite_or_nan - mean).square().nansum(-2, keepdim=True)
        var /= (finite.sum(-2, keepdim=True) - 1).clamp_(min=1)
        std = var.sqrt().clamp(min=1e-6)

        # Find values within range:
        lower = mean - self.threshold * std
        upper = mean + self.threshold * std
        keep = finite & (finite_or_nan >= lower) & (finite_or_nan <= upper)
        count = keep.sum(-2, keepdim=True)

        # Compute mean and standard deviation of kept values:
        kept_mean = torch.where(keep, finite_or_nan, 0.0).sum(-2, keepdim=True)
        kept_mean /= count.clamp(min=1)

        centered = torch.where(keep, finite_or_nan - kept_mean, 0.0)
        denominator = (count - 1).clamp(min=1)
        kept_var = centered.square().sum(-2, keepdim=True) / denominator
        kept_std = kept_var.sqrt().clamp(min=1e-6)

        has_kept = count > 0
        mean = torch.where(has_kept, kept_mean, mean)
        std = torch.where(has_kept, kept_std, std)

        self.lower_bound = mean - self.threshold * std
        self.upper_bound = mean + self.threshold * std

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        log_abs = numerical.abs().log1p()
        clipped = torch.maximum(-log_abs + self.lower_bound, numerical)
        numerical = torch.minimum(log_abs + self.upper_bound, clipped)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )
