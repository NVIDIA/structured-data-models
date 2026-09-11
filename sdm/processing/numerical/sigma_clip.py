import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ClipSigma(Processor):
    """Two-stage z-score outlier clipping with soft logarithmic bounds.

    The first pass masks values outside the initial z-score bounds, then the
    second pass refits bounds on the remaining values. The transform applies
    logarithmic soft clipping instead of hard truncation.

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
        mean = numerical.mean(dim=-2, keepdim=True)
        std = numerical.std(
            dim=-2,
            correction=1 if numerical.size(-2) > 1 else 0,
            keepdim=True,
        ).clamp_min_(1e-6)
        width = self.threshold * std
        outlier_mask = numerical < (mean - width)
        outlier_mask.logical_or_(numerical > (mean + width))

        count_dtype = (
            torch.int32 if numerical.size(-2) <= 2**31 - 1 else torch.int64
        )
        count = outlier_mask.sum(
            dim=-2,
            keepdim=True,
            dtype=count_dtype,
        )
        count.neg_().add_(numerical.size(-2))
        safe_count = count.clamp_min(1)
        centered = numerical.masked_fill(outlier_mask, 0.0)
        mean_clean = centered.sum(dim=-2, keepdim=True).div_(safe_count)
        centered.sub_(mean_clean).masked_fill_(outlier_mask, 0.0)
        del outlier_mask
        correction = (count > 1).to(count.dtype)
        denominator = (count - correction).clamp_min_(1)
        std_clean = (
            centered.square_()
            .sum(dim=-2, keepdim=True)
            .div_(denominator)
            .sqrt_()
        )

        has_clean = count > 0
        self._mean = torch.where(has_clean, mean_clean, mean)
        self._std = torch.where(has_clean, std_clean, std)
        self._std.clamp_min_(1e-6)
        width = self.threshold * self._std
        self.lower_bound = self._mean - width
        self.upper_bound = self._mean + width

    def _transform(self, table: TableTensor) -> TableTensor:
        """Clip ``table`` using the fitted soft lower and upper bounds."""
        numerical = table.numerical
        log_abs = numerical.abs().log1p_()
        clipped = self.lower_bound - log_abs
        torch.maximum(clipped, numerical, out=clipped)
        log_abs = log_abs.to(
            dtype=torch.promote_types(log_abs.dtype, self.upper_bound.dtype)
        )
        log_abs.add_(self.upper_bound)
        torch.minimum(log_abs, clipped, out=clipped)
        return table.replace_blocks(numerical=clipped)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"threshold={self.threshold})"
        )
