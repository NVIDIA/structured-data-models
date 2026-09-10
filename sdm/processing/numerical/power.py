import math

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.numerical._stats import (
    _constant_feature_mask,
    _nanmean_var,
)

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

    positive_power = ((inp * lambdas + 1).log() / lambdas).expm1()
    positive_log = inp.expm1()
    positive_out = torch.where(
        lambdas.abs() < eps,
        positive_log,
        positive_power,
    )

    two_minus_lambda = 2 - lambdas
    negative_power = -(
        (1 - two_minus_lambda * inp).log() / two_minus_lambda
    ).expm1()
    negative_log = -(-inp).expm1()
    negative_out = torch.where(
        two_minus_lambda.abs() < eps,
        negative_log,
        negative_power,
    )

    return torch.where(positive, positive_out, negative_out)


def _yeojohnson_bounds(inp: Tensor) -> tuple[Tensor, Tensor]:
    max_abs = inp.abs().max(dim=-2, keepdim=True).values
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

    all_negative = (inp < 0).all(dim=-2, keepdim=True)
    any_negative = (inp < 0).any(dim=-2, keepdim=True)

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
) -> Tensor:
    transformed = _yeojohnson_transform(
        inp,
        lambdas,
        positive_log=positive_log,
        negative_log=negative_log,
    )
    variance = transformed.var(dim=-2, correction=0, keepdim=True)
    tiny = torch.finfo(inp.dtype).tiny
    loglike = -inp.size(-2) / 2 * variance.log() + (lambdas - 1) * log_jacobian
    return torch.where(
        variance.isfinite() & (variance >= tiny),
        loglike,
        torch.full_like(loglike, -math.inf),
    )


def _optimize_lambdas(
    inp: Tensor,
    constant_features: Tensor,
) -> Tensor:
    # Reuse the sign-specific log terms across all likelihood evaluations;
    # the golden-section search only changes the per-feature lambdas.
    positive_log = inp.clamp_min(0).log1p()
    negative_log = (-inp).clamp_min(0).log1p()
    log_jacobian = torch.where(inp >= 0, positive_log, -negative_log).sum(
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
    )
    fd = _yeojohnson_log_likelihood(
        inp,
        d,
        positive_log,
        negative_log,
        log_jacobian,
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
        mean, var, count = _nanmean_var(numerical)
        filled = torch.where(numerical.isfinite(), numerical, mean)
        self.max = numerical.masked_fill(
            ~numerical.isfinite(),
            float("-inf"),
        ).amax(dim=-2, keepdim=True)
        constant_features = _constant_feature_mask(var, mean, count)
        self.lambdas = self._optimize_lambdas(filled, constant_features)

        lambda_eps = torch.finfo(numerical.dtype).eps
        self.upper_bound = -(1 / self.lambdas)
        self.upper_bound[self.lambdas > -lambda_eps] = torch.inf

        if self.standardize:
            transformed = _yeojohnson_transform(filled, self.lambdas)
            self.mean, var, _ = _nanmean_var(transformed)
            scale = var.sqrt()
            scale[_constant_feature_mask(var, self.mean, n_samples)] = 1.0
            self.scale = scale
        else:
            self.mean = torch.zeros_like(self.lambdas)
            self.scale = torch.ones_like(self.lambdas)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` with fitted Yeo-Johnson parameters."""
        numerical = table.numerical
        transformed = _yeojohnson_transform(numerical, self.lambdas)
        numerical = (transformed - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        unscaled = numerical * self.scale + self.mean
        inverse = _yeojohnson_inverse_transform(unscaled, self.lambdas)

        out_of_bounds = inverse.isinf()
        eps = torch.finfo(numerical.dtype).eps
        bounded_unscaled = torch.minimum(unscaled, self.upper_bound - eps)
        bounded_inverse = _yeojohnson_inverse_transform(
            bounded_unscaled, self.lambdas
        )
        inverse = torch.where(out_of_bounds, bounded_inverse, inverse)
        invalid = inverse.isinf()
        inverse = torch.where(
            invalid,
            torch.fmin(inverse, self.max),
            inverse,
        )

        return table.replace_blocks(numerical=inverse)
