# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        count_dtype = (
            torch.int32 if table.size(-2) <= 2**31 - 1 else torch.int64
        )

        # Compute finite mean and standard deviation:
        mean = finite_or_nan.nanmean(-2, keepdim=True)
        mean.masked_fill_(mean.isnan(), 0.0)

        var = (finite_or_nan - mean).square_().nansum(-2, keepdim=True)
        count = finite.sum(-2, keepdim=True, dtype=count_dtype)
        var /= count.sub_(1).clamp_(min=1)
        std = var.sqrt().clamp(min=1e-6)

        # Find values within range:
        width = self.threshold * std
        keep = finite_or_nan >= (mean - width)
        keep.logical_and_(finite_or_nan <= (mean + width))
        keep.logical_and_(finite)
        count = keep.sum(-2, keepdim=True, dtype=count_dtype)
        discard = ~keep

        # Compute mean and standard deviation of kept values:
        centered = finite_or_nan.masked_fill(discard, 0.0)
        kept_mean = centered.sum(-2, keepdim=True)
        kept_mean /= count.clamp(min=1)

        centered.sub_(kept_mean).masked_fill_(discard, 0.0)
        denominator = (count - 1).clamp_(min=1)
        kept_var = centered.square_().sum(-2, keepdim=True).div_(denominator)
        kept_std = kept_var.sqrt().clamp(min=1e-6)

        has_kept = count > 0
        mean = torch.where(has_kept, kept_mean, mean)
        std = torch.where(has_kept, kept_std, std)

        width = self.threshold * std
        self.lower_bound = mean - width
        self.upper_bound = mean + width

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        log_abs = numerical.abs().log1p_()
        if torch.is_grad_enabled() and (
            numerical.requires_grad
            or self.lower_bound.requires_grad
            or self.upper_bound.requires_grad
        ):
            clipped = torch.maximum(self.lower_bound - log_abs, numerical)
            clipped = torch.minimum(log_abs + self.upper_bound, clipped)
            return table.replace_blocks(numerical=clipped)

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
