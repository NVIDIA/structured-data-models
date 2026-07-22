import math
from collections.abc import Callable

import torch
from torch import Tensor

from sdm.processing._utils import _as_float
from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.numerical._stats import _constant_feature_mask
from sdm.stype import Stype
from sdm.tensor import TableTensor


def _yeojohnson_transform_col(inp: Tensor, lmbda: float) -> Tensor:
    output = torch.zeros_like(inp)
    positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps

    if abs(lmbda) < eps:
        output[positive] = inp[positive].log1p()
    else:
        output[positive] = (lmbda * inp[positive].log1p()).expm1() / lmbda

    if abs(lmbda - 2) > eps:
        output[~positive] = -(
            (2 - lmbda) * (-inp[~positive]).log1p()
        ).expm1() / (2 - lmbda)
    else:
        output[~positive] = -(-inp[~positive]).log1p()

    return output


def _yeojohnson_transform(inp: Tensor, lambdas: Tensor) -> Tensor:
    lambdas = lambdas.unsqueeze(0)
    eps = torch.finfo(inp.dtype).eps

    positive_log = inp.clamp_min(0).log1p()
    positive = (lambdas * positive_log).expm1() / lambdas
    positive = torch.where(lambdas.abs() < eps, positive_log, positive)

    two_minus_lambda = 2 - lambdas
    negative_log = (-inp).clamp_min(0).log1p()
    negative = -((two_minus_lambda * negative_log).expm1() / two_minus_lambda)
    negative = torch.where(
        two_minus_lambda.abs() < eps,
        -negative_log,
        negative,
    )

    return torch.where(inp >= 0, positive, negative)


def _yeojohnson_inverse_transform(inp: Tensor, lmbda: float) -> Tensor:
    inverse = torch.zeros_like(inp)
    positive = inp >= 0
    eps = torch.finfo(inp.dtype).eps

    if abs(lmbda) < eps:
        inverse[positive] = inp[positive].expm1()
    else:
        inverse[positive] = ((inp[positive] * lmbda + 1).log() / lmbda).expm1()

    if abs(lmbda - 2) > eps:
        inverse[~positive] = -(
            (-(2 - lmbda) * inp[~positive] + 1).log() / (2 - lmbda)
        ).expm1()
    else:
        inverse[~positive] = -(-inp[~positive]).expm1()

    return inverse


def _yeojohnson_log_likelihood(inp: Tensor, lmbda: float) -> float:
    transformed = _yeojohnson_transform_col(inp, lmbda)
    variance = transformed.var(correction=0)
    if not variance.isfinite() or variance < torch.finfo(variance.dtype).tiny:
        return -math.inf

    loglike = -inp.numel() / 2 * variance.log() + (lmbda - 1) * torch.sum(
        inp.sign() * inp.abs().log1p()
    )
    return float(loglike)


def _yeojohnson_bounds(inp: Tensor) -> tuple[float, float]:
    max_abs = inp.abs().max()
    if max_abs == 0:
        return 1.0, 1.0

    log1p_max_x = torch.log1p(20 * max_abs).item()
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny_float = (math.log(finfo.tiny) - log_eps) / 2
    log_max_float = (math.log(finfo.max) + log_eps) / 2

    lower_bound = log_tiny_float / log1p_max_x
    upper_bound = log_max_float / log1p_max_x
    if torch.all(inp < 0):
        lower_bound, upper_bound = 2 - upper_bound, 2 - lower_bound
    elif torch.any(inp < 0):
        lower_bound = max(2 - upper_bound, lower_bound)
        upper_bound = min(2 - lower_bound, upper_bound)

    return lower_bound, upper_bound


def _bounded_argmax(
    function: Callable[[float], float],
    lower_bound: float,
    upper_bound: float,
    *,
    xatol: float = 1.48e-8,
    max_iter: int = 500,
) -> float:
    invphi = (math.sqrt(5) - 1) / 2
    left = lower_bound
    right = upper_bound
    c = right - invphi * (right - left)
    d = left + invphi * (right - left)
    fc = function(c)
    fd = function(d)

    for _ in range(max_iter):
        if abs(right - left) <= xatol:
            break
        if fc < fd:
            left = c
            c = d
            fc = fd
            d = left + invphi * (right - left)
            fd = function(d)
        else:
            right = d
            d = c
            fd = fc
            c = right - invphi * (right - left)
            fc = function(c)

    return (left + right) / 2


def _nan_mean_var(inp: Tensor) -> tuple[Tensor, Tensor]:
    finite = ~inp.isnan()
    counts = finite.sum(dim=0)
    safe_counts = counts.clamp(min=1)
    finite_input = torch.where(finite, inp, torch.zeros_like(inp))
    mean = finite_input.sum(dim=0) / safe_counts
    centered = torch.where(finite, inp - mean, torch.zeros_like(inp))
    var = (centered * centered).sum(dim=0) / safe_counts

    nan = torch.full_like(mean, torch.nan)
    mean = torch.where(counts > 0, mean, nan)
    var = torch.where(counts > 0, var, nan)
    return mean, var


def _nan_max(inp: Tensor) -> Tensor:
    """Per-column max over finite values; all-NaN columns map to NaN."""
    finite = ~inp.isnan()
    counts = finite.sum(dim=0)
    neg_inf = torch.full_like(inp, float("-inf"))
    result = torch.where(finite, inp, neg_inf).max(dim=0).values

    nan = torch.full_like(result, torch.nan)
    return torch.where(counts > 0, result, nan)


class Power(Processor, InvertibleMixin):
    """Apply a feature-wise Yeo-Johnson power transform.

    Args:
        standardize: If ``True``, zero-mean and unit-variance the transformed
            features using statistics fitted after the power transform.
    """

    supported_stypes = frozenset({Stype.numerical})

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

    def _optimize_lambda(self, inp: Tensor) -> float:
        finite = inp[inp.isfinite()]
        if finite.numel() < 2 or torch.all(finite == finite[0]):
            return 1.0

        lower_bound, upper_bound = _yeojohnson_bounds(finite)
        if lower_bound == upper_bound:
            return lower_bound

        return _bounded_argmax(
            lambda lmbda: _yeojohnson_log_likelihood(finite, lmbda),
            lower_bound,
            upper_bound,
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = _as_float(table.numerical)
        n_samples, n_features = numerical.shape

        var = numerical.var(dim=0, correction=0)
        mean = numerical.mean(dim=0)
        self.max = _nan_max(numerical)
        lambdas = numerical.new_empty(n_features)

        constant_features = _constant_feature_mask(var, mean, n_samples)
        for i in range(n_features):
            col = numerical[:, i]
            lmbda = 1.0 if constant_features[i] else self._optimize_lambda(col)
            lambdas[i] = lmbda

        self.lambdas = lambdas

        lambda_eps = torch.finfo(numerical.dtype).eps
        self.upper_bound = -(1 / self.lambdas)
        self.upper_bound[self.lambdas > -lambda_eps] = torch.inf

        if self.standardize:
            transformed = _yeojohnson_transform(numerical, self.lambdas)
            self.mean, var = _nan_mean_var(transformed)
            scale = var.sqrt()
            scale[_constant_feature_mask(var, self.mean, n_samples)] = 1.0
            self.scale = scale
        else:
            self.mean = numerical.new_zeros(n_features)
            self.scale = numerical.new_ones(n_features)

    def _yeojohnson_inverse_transform(self, inp: Tensor) -> Tensor:
        inverse = inp.clone()
        for i, lmbda in enumerate(self.lambdas):
            inverse[:, i] = _yeojohnson_inverse_transform(
                inp[:, i],
                float(lmbda),
            )
        return inverse

    def _transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` with fitted Yeo-Johnson parameters."""
        numerical = _as_float(table.numerical)
        transformed = _yeojohnson_transform(numerical, self.lambdas)
        numerical = (transformed - self.mean) / self.scale
        return table.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        numerical = _as_float(table.numerical)
        unscaled = numerical * self.scale + self.mean
        inverse = self._yeojohnson_inverse_transform(unscaled)

        out_of_bounds = inverse.isinf()
        if out_of_bounds.any():
            eps = torch.finfo(numerical.dtype).eps
            unscaled = torch.minimum(unscaled, self.upper_bound - eps)
            inverse[out_of_bounds] = self._yeojohnson_inverse_transform(
                unscaled,
            )[out_of_bounds]
            invalid = inverse.isinf()
            inverse[invalid] = torch.fmin(inverse, self.max)[invalid]

        return table.replace_blocks(numerical=inverse)
