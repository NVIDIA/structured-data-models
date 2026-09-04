import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _constant_feature_mask


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    Constant columns use a unit scale to keep the transform finite and
    invertible.

    Args:
        with_mean: If ``True``, center each column by its fitted mean.
        with_std: If ``True``, scale each column by its fitted standard
            deviation.
        epsilon: Value added to each fitted standard deviation. The default
            preserves exact constant-column handling.
        ignore_nan: If ``True``, fit statistics from non-NaN values and
            preserve NaNs during transformation. All-missing columns use zero
            mean and unit scale.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        epsilon: float = 0.0,
        ignore_nan: bool = False,
    ) -> None:
        super().__init__()
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = epsilon
        self.ignore_nan = ignore_nan
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        if numerical.size(-1) == 0:
            self.mean = numerical.sum(dim=-2, keepdim=True)
            self.scale = torch.ones_like(self.mean)
            return

        count: int | torch.Tensor = numerical.size(-2)
        if self.ignore_nan:
            observed = ~numerical.isnan()
            count = observed.sum(dim=-2, keepdim=True)
            data_mean = numerical.masked_fill(~observed, 0.0).sum(
                dim=-2,
                keepdim=True,
            )
            data_mean = data_mean / count.clamp(min=1)
        else:
            observed = None
            data_mean = numerical.mean(dim=-2, keepdim=True)

        if self.with_mean:
            self.mean = data_mean
        else:
            self.mean = torch.zeros_like(data_mean)

        if self.with_std:
            if numerical.size(-2) > 1:
                if observed is None:
                    var = numerical.var(
                        dim=-2,
                        correction=0,
                        keepdim=True,
                    )
                else:
                    assert isinstance(count, torch.Tensor)
                    centered = (numerical - data_mean).masked_fill(
                        ~observed,
                        0.0,
                    )
                    var = centered.square().sum(dim=-2, keepdim=True)
                    var = var / count.clamp(min=1)
                scale = var.sqrt()
                if self.epsilon == 0:
                    scale[
                        _constant_feature_mask(
                            var,
                            data_mean,
                            count,
                        )
                    ] = 1.0
            else:
                if self.epsilon == 0:
                    scale = torch.ones_like(data_mean)
                else:
                    scale = torch.zeros_like(data_mean)
            self.scale = scale + self.epsilon
            if self.ignore_nan:
                assert isinstance(count, torch.Tensor)
                self.scale = self.scale.masked_fill(count == 0, 1.0)
        else:
            self.scale = torch.ones_like(data_mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` using the fitted mean and scale."""
        numerical = (table.numerical - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical * self.scale + self.mean
        return table.replace_blocks(numerical=numerical)
