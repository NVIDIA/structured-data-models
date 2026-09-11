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
    magnitude_log: Tensor | None = None,
    positive: Tensor | None = None,
    exponents: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    if magnitude_log is None:
        magnitude_log = inp.abs().log1p_()
    if positive is None:
        positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps
    two_minus_lambda = 2 - lambdas
    exponents = torch.where(positive, lambdas, two_minus_lambda, out=exponents)
    out = torch.mul(exponents, magnitude_log, out=out)
    out.expm1_().div_(exponents)
    zero_exponent = torch.where(
        positive, lambdas.abs() < eps, two_minus_lambda.abs() < eps
    )
    torch.where(zero_exponent, magnitude_log, out, out=out)
    return out.copysign_(inp)


def _yeojohnson_inverse_transform(inp: Tensor, lambdas: Tensor) -> Tensor:
    positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps
    two_minus_lambda = 2 - lambdas
    exponents = torch.where(positive, lambdas, two_minus_lambda)
    magnitude = inp.abs()
    inverse = (magnitude * exponents).add_(1).log_().div_(exponents).expm1_()
    del exponents
    zero_exponent = torch.where(
        positive, lambdas.abs() < eps, two_minus_lambda.abs() < eps
    )
    torch.where(zero_exponent, magnitude.expm1_(), inverse, out=inverse)
    return inverse.copysign_(inp)


def _yeojohnson_bounds(inp: Tensor) -> tuple[Tensor, Tensor]:
    minimum, maximum = torch.aminmax(inp, dim=-2, keepdim=True)
    max_abs = torch.maximum(minimum.neg(), maximum)
    log1p_max_x = (20 * max_abs).log1p_()
    log1p_max_x.masked_fill_(max_abs == 0, 1.0)
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny_float = (math.log(finfo.tiny) - log_eps) / 2
    log_max_float = (math.log(finfo.max) + log_eps) / 2

    lower_bound = log_tiny_float / log1p_max_x
    upper_bound = log_max_float / log1p_max_x
    positive_lower = lower_bound
    positive_upper = upper_bound

    all_negative = maximum < 0
    any_negative = minimum < 0

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
        lower_bound.masked_fill_(zero, 1.0),
        upper_bound.masked_fill_(zero, 1.0),
    )


def _yeojohnson_log_likelihood(
    inp: Tensor,
    lambdas: Tensor,
    magnitude_log: Tensor,
    positive: Tensor,
    log_jacobian: Tensor,
    exponents: Tensor,
    transformed: Tensor,
) -> Tensor:
    transformed = _yeojohnson_transform(
        inp,
        lambdas,
        magnitude_log=magnitude_log,
        positive=positive,
        exponents=exponents,
        out=transformed,
    )
    variance = transformed.var(dim=-2, correction=0, keepdim=True)
    tiny = torch.finfo(inp.dtype).tiny
    valid = variance.isfinite() & (variance >= tiny)
    loglike = variance.log_().mul_(-inp.size(-2) / 2)
    loglike.add_((lambdas - 1).mul_(log_jacobian))
    return loglike.masked_fill_(~valid, -math.inf)


def _optimize_lambdas(
    inp: Tensor,
    constant_features: Tensor,
) -> Tensor:
    # Both signs use the same log magnitude with a different exponent.
    # Reuse the full-table workspaces throughout the golden-section search.
    magnitude_log = inp.abs().log1p_()
    positive = inp >= 0
    log_jacobian = magnitude_log.copysign(inp).sum(
        dim=-2,
        keepdim=True,
    )
    exponents = torch.empty_like(inp)
    transformed = torch.empty_like(inp)

    left, right = _yeojohnson_bounds(inp)
    left.masked_fill_(constant_features, 1.0)
    right.masked_fill_(constant_features, 1.0)

    invphi = (math.sqrt(5) - 1) / 2
    span = (right - left).mul_(invphi)
    c = right - span
    d = left + span
    fc = _yeojohnson_log_likelihood(
        inp,
        c,
        magnitude_log,
        positive,
        log_jacobian,
        exponents,
        transformed,
    )
    fd = _yeojohnson_log_likelihood(
        inp,
        d,
        magnitude_log,
        positive,
        log_jacobian,
        exponents,
        transformed,
    )
    c_next = torch.empty_like(c)
    d_next = torch.empty_like(d)
    new_point = torch.empty_like(c)
    choose_right = torch.empty_like(c, dtype=torch.bool)

    for _ in range(_YEOJOHNSON_OPTIMIZATION_STEPS):
        torch.lt(fc, fd, out=choose_right)
        # Keep the search fully vectorized: each feature independently
        # chooses its next interval without per-column Python branching.
        torch.where(choose_right, c, left, out=left)
        torch.where(choose_right, right, d, out=right)
        torch.sub(right, left, out=span).mul_(invphi)
        torch.sub(right, span, out=c_next)
        torch.where(choose_right, d, c_next, out=c_next)
        torch.add(left, span, out=d_next)
        torch.where(choose_right, d_next, c, out=d_next)
        torch.where(choose_right, d_next, c_next, out=new_point)
        new_score = _yeojohnson_log_likelihood(
            inp,
            new_point,
            magnitude_log,
            positive,
            log_jacobian,
            exponents,
            transformed,
        )
        torch.where(choose_right, new_score, fc, out=fc)
        torch.where(choose_right, fd, new_score, out=fd)
        fc, fd = fd, fc
        c, c_next = c_next, c
        d, d_next = d_next, d

    lambdas = left.add_(right).div_(2)
    return lambdas.masked_fill_(constant_features, 1.0)


class PowerTransform(Processor, InvertibleMixin):
    """Apply a feature-wise Yeo-Johnson power transform.

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
        self.register_buffer("upper_bound", torch.empty(0))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    def _optimize_lambdas(
        self,
        inp: Tensor,
        constant_features: Tensor,
    ) -> Tensor:
        return _optimize_lambdas(inp, constant_features)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        n_samples = numerical.size(-2)

        var = numerical.var(dim=-2, correction=0, keepdim=True)
        mean = numerical.mean(dim=-2, keepdim=True)
        self.max = numerical.max(dim=-2, keepdim=True).values
        constant_features = _constant_feature_mask(var, mean, n_samples)
        self.lambdas = self._optimize_lambdas(numerical, constant_features)

        lambda_eps = torch.finfo(numerical.dtype).eps
        self.upper_bound = self.lambdas.reciprocal().neg_()
        self.upper_bound.masked_fill_(self.lambdas > -lambda_eps, torch.inf)

        if self.standardize:
            transformed = _yeojohnson_transform(numerical, self.lambdas)
            self.mean = transformed.mean(dim=-2, keepdim=True)
            var = transformed.var(dim=-2, correction=0, keepdim=True)
            constant_features = _constant_feature_mask(
                var, self.mean, n_samples
            )
            self.scale = var.sqrt_().masked_fill_(constant_features, 1.0)
        else:
            self.mean = torch.zeros_like(self.lambdas)
            self.scale = torch.ones_like(self.lambdas)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` with fitted Yeo-Johnson parameters."""
        numerical = table.numerical
        transformed = _yeojohnson_transform(numerical, self.lambdas)
        numerical = transformed.sub_(self.mean).div_(self.scale)
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        unscaled = (numerical * self.scale).add_(self.mean)
        inverse = _yeojohnson_inverse_transform(unscaled, self.lambdas)

        out_of_bounds = inverse.isposinf()
        eps = torch.finfo(numerical.dtype).eps
        # Only the positive branch with lambda <= -eps has a finite upper
        # bound. Reuse the unscaled input for that overflow correction.
        out_of_bounds.logical_and_(self.lambdas <= -eps)
        unscaled.clamp_max_(self.upper_bound - eps)
        unscaled.mul_(self.lambdas).add_(1).log_().div_(self.lambdas).expm1_()
        torch.where(out_of_bounds, unscaled, inverse, out=inverse)
        torch.isposinf(inverse, out=out_of_bounds)
        out_of_bounds.logical_and_(~self.max.isnan())
        torch.where(out_of_bounds, self.max, inverse, out=inverse)

        return table.replace_blocks(numerical=inverse)
