import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _nanmean_var,
)


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible. NaN and infinite values are ignored when fitting statistics
    and preserved during the transform.

    Args:
        with_mean: If ``True``, center each column by its fitted mean.
        with_std: If ``True``, scale each column by its fitted standard
            deviation.
        epsilon: Value added to each fitted standard deviation. The default
            preserves exact constant-column handling.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = epsilon
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        if numerical.size(-1) == 0 or not (self.with_mean or self.with_std):
            self.mean = numerical.new_zeros(
                (*numerical.shape[:-2], 1, numerical.size(-1))
            )
            self.scale = torch.ones_like(self.mean)
            return

        finite = numerical.isfinite()
        finite_or_nan = numerical.masked_fill(~finite, float("nan"))
        mean = finite_or_nan.nanmean(dim=-2, keepdim=True)
        mean = torch.where(mean.isnan(), torch.zeros_like(mean), mean)

        if self.with_mean:
            self.mean = mean
        else:
            self.mean = torch.zeros_like(mean)

        if self.with_std:
            if numerical.size(-2) > 1:
                finite_or_nan.sub_(mean).square_()
                var = finite_or_nan.nanmean(dim=-2, keepdim=True)
                var.masked_fill_(var.isnan(), 0.0)
                scale = var.sqrt()
                if self.epsilon == 0:
                    scale[
                        _constant_feature_mask(
                            var,
                            mean,
                            finite.sum(dim=-2, keepdim=True),
                        )
                    ] = 1.0
            else:
                if self.epsilon == 0:
                    scale = torch.ones_like(mean)
                else:
                    scale = torch.zeros_like(mean)
            self.scale = scale + self.epsilon
        else:
            self.scale = torch.ones_like(mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` using the fitted mean and scale."""
        numerical = (table.numerical - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical * self.scale + self.mean
        return table.replace_blocks(numerical=numerical)
