from typing import Literal

import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _constant_feature_mask


class Standardize(Processor, InvertibleMixin):
    """Center and scale each feature column.

    By default, constant columns use a unit scale to keep the transform finite
    and invertible.

    Args:
        with_mean: If ``True``, center each column by its fitted mean.
        with_std: If ``True``, scale each column by its fitted standard
            deviation.
        correction: Use ``0`` for population variance or ``1`` for sample
            variance.
        constant_threshold: Standard deviations below this value are replaced
            by one before adding ``epsilon``. If ``None``, preserve the default
            constant-feature handling.
        min_scale: Minimum fitted scale, overriding the default constant-column
            handling. If ``None``, do not clamp the scale.
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
        correction: Literal[0, 1] = 0,
        constant_threshold: float | None = None,
        min_scale: float | None = None,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        if constant_threshold is not None and constant_threshold <= 0:
            raise ValueError("constant_threshold must be positive.")
        if min_scale is not None and min_scale <= 0:
            raise ValueError("min_scale must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.with_mean = with_mean
        self.with_std = with_std
        self.correction = correction
        self.constant_threshold = constant_threshold
        self.min_scale = min_scale
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
        if numerical.size(-1) == 0:
            self.mean = numerical.sum(dim=-2, keepdim=True)
            self.scale = torch.ones_like(self.mean)
            return

        data_mean = numerical.mean(dim=-2, keepdim=True)

        if self.with_mean:
            self.mean = data_mean
        else:
            self.mean = torch.zeros_like(data_mean)

        if self.with_std:
            if numerical.size(-2) > 1:
                var = numerical.var(
                    dim=-2,
                    correction=self.correction,
                    keepdim=True,
                )
                scale = var.sqrt()
                if self.constant_threshold is not None:
                    scale = scale.masked_fill(
                        scale < self.constant_threshold,
                        1.0,
                    )
                elif self.min_scale is None and self.epsilon == 0:
                    scale[
                        _constant_feature_mask(
                            var,
                            data_mean,
                            numerical.size(-2),
                        )
                    ] = 1.0
            else:
                if self.constant_threshold is not None or (
                    self.min_scale is None and self.epsilon == 0
                ):
                    scale = torch.ones_like(data_mean)
                else:
                    scale = torch.zeros_like(data_mean)
            if self.min_scale is not None:
                scale = scale.clamp_min(self.min_scale)
            self.scale = scale + self.epsilon
        else:
            self.scale = torch.ones_like(data_mean)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` using the fitted mean and scale."""
        numerical = (table.numerical - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical * self.scale + self.mean
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if not self.with_mean:
            arguments.append("with_mean=False")
        if not self.with_std:
            arguments.append("with_std=False")
        if self.correction != 0:
            arguments.append(f"correction={self.correction}")
        if self.constant_threshold is not None:
            arguments.append(f"constant_threshold={self.constant_threshold!r}")
        if self.min_scale is not None:
            arguments.append(f"min_scale={self.min_scale!r}")
        if self.epsilon != 0.0:
            arguments.append(f"epsilon={self.epsilon!r}")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )
