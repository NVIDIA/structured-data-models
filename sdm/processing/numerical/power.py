# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _constant_feature_mask

# Keep GPU execution batched; adaptive per-column stopping would resynchronize.
# For float32 overflow-safe bounds, 44 golden steps reaches ~1.48e-8.
_YEOJOHNSON_OPTIMIZATION_STEPS = 44


def _yeojohnson_transform(
    inp: Tensor,
    lambdas: Tensor,
    *,
    positive_log: Tensor | None = None,
    negative_log: Tensor | None = None,
) -> Tensor:
    if positive_log is None:
        positive_log = inp.clamp_min(0).log1p()
    if negative_log is None:
        negative_log = (-inp).clamp_min(0).log1p()

    eps = torch.finfo(inp.dtype).eps

    positive = (lambdas * positive_log).expm1() / lambdas
    positive = torch.where(lambdas.abs() < eps, positive_log, positive)

    two_minus_lambda = 2 - lambdas
    negative = -((two_minus_lambda * negative_log).expm1() / two_minus_lambda)
    negative = torch.where(
        two_minus_lambda.abs() < eps,
        -negative_log,
        negative,
    )

    return torch.where(inp >= 0, positive, negative)


def _yeojohnson_inverse_transform(inp: Tensor, lambdas: Tensor) -> Tensor:
    positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps

    # Mirror the ``log1p``/``expm1`` pair of the forward transform: a small
    # ``inp * lambdas`` loses its precision in ``(1 + x).log()``.
    positive_power = ((inp * lambdas).log1p() / lambdas).expm1()
    positive_log = inp.expm1()
    positive_out = torch.where(
        lambdas.abs() < eps,
        positive_log,
        positive_power,
    )

    two_minus_lambda = 2 - lambdas
    negative_power = -(
        (-two_minus_lambda * inp).log1p() / two_minus_lambda
    ).expm1()
    negative_log = -(-inp).expm1()
    negative_out = torch.where(
        two_minus_lambda.abs() < eps,
        negative_log,
        negative_power,
    )

    return torch.where(positive, positive_out, negative_out)


def _yeojohnson_bounds(inp: Tensor) -> tuple[Tensor, Tensor]:
    missing = inp.isnan()
    max_abs = inp.abs().nan_to_num(nan=0.0).amax(dim=-2, keepdim=True)
    log1p_max_x = (20 * max_abs).log1p()
    log1p_max_x = torch.where(
        max_abs == 0,
        torch.ones_like(log1p_max_x),
        log1p_max_x,
    )
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny_float = (math.log(finfo.tiny) - log_eps) / 2
    log_max_float = (math.log(finfo.max) + log_eps) / 2

    lower_bound = log_tiny_float / log1p_max_x
    upper_bound = log_max_float / log1p_max_x
    positive_lower = lower_bound
    positive_upper = upper_bound

    negative = inp < 0
    all_negative = (negative | missing).all(dim=-2, keepdim=True)
    any_negative = negative.any(dim=-2, keepdim=True)

    mixed_lower = torch.maximum(2 - positive_upper, positive_lower)
    mixed_upper = torch.minimum(2 - mixed_lower, positive_upper)
    lower_bound = torch.where(any_negative, mixed_lower, positive_lower)
    upper_bound = torch.where(any_negative, mixed_upper, positive_upper)

    negative_lower = 2 - positive_upper
    negative_upper = 2 - positive_lower
    lower_bound = torch.where(all_negative, negative_lower, lower_bound)
    upper_bound = torch.where(all_negative, negative_upper, upper_bound)

    zero = max_abs == 0
    return (
        torch.where(zero, torch.ones_like(lower_bound), lower_bound),
        torch.where(zero, torch.ones_like(upper_bound), upper_bound),
    )


def _yeojohnson_log_likelihood(
    inp: Tensor,
    lambdas: Tensor,
    positive_log: Tensor,
    negative_log: Tensor,
    log_jacobian: Tensor,
    count: Tensor,
) -> Tensor:
    transformed = _yeojohnson_transform(
        inp,
        lambdas,
        positive_log=positive_log,
        negative_log=negative_log,
    )
    mean = transformed.nanmean(dim=-2, keepdim=True)
    variance = (transformed - mean).square().nanmean(dim=-2, keepdim=True)
    tiny = torch.finfo(inp.dtype).tiny
    loglike = -count / 2 * variance.log() + (lambdas - 1) * log_jacobian
    return torch.where(
        variance.isfinite() & (variance >= tiny),
        loglike,
        torch.full_like(loglike, -math.inf),
    )


def _optimize_lambdas(
    inp: Tensor,
    constant_features: Tensor,
    *,
    count: Tensor,
) -> Tensor:
    # Reuse the sign-specific log terms across all likelihood evaluations;
    # the golden-section search only changes the per-feature lambdas.
    positive_log = inp.clamp_min(0).log1p()
    negative_log = (-inp).clamp_min(0).log1p()
    log_jacobian = torch.where(inp >= 0, positive_log, -negative_log).nansum(
        dim=-2,
        keepdim=True,
    )

    left, right = _yeojohnson_bounds(inp)
    identity = torch.ones_like(left)
    left = torch.where(constant_features, identity, left)
    right = torch.where(constant_features, identity, right)

    invphi = (math.sqrt(5) - 1) / 2
    c = right - invphi * (right - left)
    d = left + invphi * (right - left)
    fc = _yeojohnson_log_likelihood(
        inp,
        c,
        positive_log,
        negative_log,
        log_jacobian,
        count,
    )
    fd = _yeojohnson_log_likelihood(
        inp,
        d,
        positive_log,
        negative_log,
        log_jacobian,
        count,
    )

    for _ in range(_YEOJOHNSON_OPTIMIZATION_STEPS):
        choose_right = fc < fd
        old_fc = fc
        old_fd = fd
        # Keep the search fully vectorized: each feature independently
        # chooses its next interval without per-column Python branching.
        left_next = torch.where(choose_right, c, left)
        right_next = torch.where(choose_right, right, d)
        c_next = torch.where(
            choose_right,
            d,
            right_next - invphi * (right_next - left_next),
        )
        d_next = torch.where(
            choose_right,
            left_next + invphi * (right_next - left_next),
            c,
        )
        new_point = torch.where(choose_right, d_next, c_next)
        new_score = _yeojohnson_log_likelihood(
            inp,
            new_point,
            positive_log,
            negative_log,
            log_jacobian,
            count,
        )
        fc = torch.where(choose_right, old_fd, new_score)
        fd = torch.where(choose_right, new_score, old_fc)
        left = left_next
        right = right_next
        c = c_next
        d = d_next

    lambdas = (left + right) / 2
    return torch.where(constant_features, identity, lambdas)


class PowerTransform(Processor, InvertibleMixin):
    """Apply a feature-wise Yeo-Johnson power transform.

    NaN and infinite values are left out of the fitted statistics. NaN values
    are preserved during the transform.

    Args:
        standardize: If ``True``, zero-mean and unit-variance the transformed
            features using statistics fitted after the power transform.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        *,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        self.standardize = standardize
        self.register_buffer("lambdas", torch.empty(0))
        self.register_buffer("max", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _optimize_lambdas(
        self,
        inp: Tensor,
        constant_features: Tensor,
        *,
        count: Tensor,
    ) -> Tensor:
        return _optimize_lambdas(inp, constant_features, count=count)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        finite = table.numerical.isfinite()
        finite_or_nan = table.numerical.masked_fill(~finite, torch.nan)
        count = finite.sum(dim=-2, keepdim=True)

        mean = finite_or_nan.nanmean(-2, keepdim=True)
        mean.masked_fill_(mean.isnan(), 0.0)
        var = (finite_or_nan - mean).square().nanmean(-2, keepdim=True)
        var.masked_fill_(var.isnan(), 0.0)
        constant_features = _constant_feature_mask(
            var,
            mean,
            num_samples=count,
        )
        del var

        # ``mean`` never exceeds the column maximum, and is zero for an
        # entirely missing column.
        self.max = torch.where(finite, table.numerical, mean).amax(
            dim=-2,
            keepdim=True,
        )
        del finite

        self.lambdas = self._optimize_lambdas(
            finite_or_nan,
            constant_features,
            count=count,
        )

        if self.standardize:
            transformed = _yeojohnson_transform(finite_or_nan, self.lambdas)
            mean = transformed.nanmean(dim=-2, keepdim=True)
            mean.masked_fill_(mean.isnan(), 0.0)
            var = (transformed - mean).square().nanmean(dim=-2, keepdim=True)
            var.masked_fill_(var.isnan(), 0.0)
            scale = var.sqrt()
            scale[_constant_feature_mask(var, mean, count)] = 1.0
            self.mean = mean
            self.scale = scale
        else:
            self.mean = torch.zeros_like(self.lambdas)
            self.scale = torch.ones_like(self.lambdas)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` with fitted Yeo-Johnson parameters."""
        transformed = _yeojohnson_transform(table.numerical, self.lambdas)
        numerical = (transformed - self.mean) / self.scale
        # The fitted lambdas only keep the fitted range representable, so a
        # query far outside it can still overflow.
        bound = torch.finfo(numerical.dtype).max
        return table.replace_blocks(
            numerical=numerical.clamp(min=-bound, max=bound)
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        unscaled = table.numerical * self.scale + self.mean
        inverse = _yeojohnson_inverse_transform(unscaled, self.lambdas)

        # Above the fitted upper bound the inverse diverges, either to
        # infinity or, past the asymptote, to NaN. Fall back to the largest
        # fitted value and keep missing values missing.
        diverged = ~inverse.isfinite() & ~unscaled.isnan()
        return table.replace_blocks(
            numerical=torch.where(
                diverged,
                torch.fmin(inverse, self.max),
                inverse,
            )
        )
