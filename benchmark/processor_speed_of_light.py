"""Measure practical GPU lower bounds for slow processing operations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from sdm import Stype, TableTensor
from sdm.processing import (
    CategoricalAlign,
    CategoryShuffle,
    Clip,
    FeaturePermute,
    Power,
    Quantile,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
)
from sdm.processing.sigma_clip import _nanstd
from torch import Tensor

from benchmark.tabiclv2_processing import (
    Characteristics,
    build_workload,
)

POWER_LAMBDA_ATOL = 1e-3
POWER_OUTPUT_ATOL = 5e-3
QUANTILE_OUTPUT_ATOL = 2e-5
YJ_SEARCH_ATOL = 1.48e-8
YJ_SEARCH_ITERATIONS = 48


@dataclass(frozen=True)
class Timing:
    """Host and CUDA timing for one already-prepared operation."""

    median_ms: float
    p95_ms: float
    cuda_median_ms: float
    cuda_p95_ms: float
    host_overhead_median_ms: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class Result:
    """One speed-of-light benchmark result."""

    processor: str
    operation: str
    candidate: str
    dtype: str
    input_shape: tuple[int, ...]
    dataset_characteristics: str
    median_ms: float
    p95_ms: float
    cuda_median_ms: float
    cuda_p95_ms: float
    host_overhead_median_ms: float
    peak_memory_bytes: int
    throughput_values_per_second: float
    max_abs_error: float | None
    correctness_status: str
    repetitions: int
    warmups: int
    gpu_model: str


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _measure_cuda(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    repetitions: int,
    warmups: int,
) -> Timing:
    for _ in range(warmups):
        result = operation()
        torch.cuda.synchronize(device)
        del result

    wall_times: list[float] = []
    cuda_times: list[float] = []
    host_overheads: list[float] = []
    peak_memory = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        wall_start = time.perf_counter_ns()
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        cuda_ms = start.elapsed_time(end)

        wall_times.append(wall_ms)
        cuda_times.append(cuda_ms)
        host_overheads.append(max(0.0, wall_ms - cuda_ms))
        peak_memory = max(
            peak_memory,
            max(
                0,
                torch.cuda.max_memory_allocated(device) - baseline_memory,
            ),
        )
        del result

    return Timing(
        median_ms=statistics.median(wall_times),
        p95_ms=_p95(wall_times),
        cuda_median_ms=statistics.median(cuda_times),
        cuda_p95_ms=_p95(cuda_times),
        host_overhead_median_ms=statistics.median(host_overheads),
        peak_memory_bytes=peak_memory,
    )


def _vectorized_yeojohnson(inp: Tensor, lambdas: Tensor) -> Tensor:
    lambdas = lambdas.to(dtype=inp.dtype).unsqueeze(0)
    eps = torch.finfo(inp.dtype).eps

    positive_log = inp.clamp_min(0).log1p()
    positive = (lambdas * positive_log).expm1() / lambdas
    positive = torch.where(lambdas.abs() < eps, positive_log, positive)

    negative_log = (-inp).clamp_min(0).log1p()
    two_minus_lambda = 2 - lambdas
    negative = -((two_minus_lambda * negative_log).expm1() / two_minus_lambda)
    negative = torch.where(
        two_minus_lambda.abs() < eps,
        -negative_log,
        negative,
    )
    return torch.where(inp >= 0, positive, negative)


def _power_log_likelihood(inp: Tensor, lambdas: Tensor) -> Tensor:
    finite = inp.isfinite()
    counts = finite.sum(dim=0)
    safe_counts = counts.clamp(min=1)
    clean = torch.where(finite, inp, torch.zeros_like(inp))
    transformed = _vectorized_yeojohnson(clean, lambdas)
    transformed = torch.where(finite, transformed, torch.zeros_like(inp))
    mean = transformed.sum(dim=0) / safe_counts
    centered = torch.where(
        finite,
        transformed - mean,
        torch.zeros_like(inp),
    )
    variance = (centered * centered).sum(dim=0) / safe_counts
    jacobian = (clean.sign() * clean.abs().log1p()).sum(dim=0)
    typed_lambdas = lambdas.to(dtype=inp.dtype)
    loglike = -counts / 2 * variance.log() + (typed_lambdas - 1) * jacobian
    invalid = (counts < 2) | ~variance.isfinite()
    invalid |= variance < torch.finfo(inp.dtype).tiny
    return torch.where(invalid, -torch.inf, loglike)


def _power_bounds(inp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    finite = inp.isfinite()
    counts = finite.sum(dim=0)
    clean = torch.where(finite, inp, torch.zeros_like(inp))
    max_abs = clean.abs().max(dim=0).values

    log1p_max = (20 * max_abs).log1p()
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny = (math.log(finfo.tiny) - log_eps) / 2
    log_max = (math.log(finfo.max) + log_eps) / 2
    lower = (log_tiny / log1p_max).to(torch.float64)
    upper = (log_max / log1p_max).to(torch.float64)

    all_negative = ((inp < 0) | ~finite).all(dim=0) & (counts > 0)
    any_negative = ((inp < 0) & finite).any(dim=0)
    negative_lower = 2 - upper
    negative_upper = 2 - lower
    mixed_lower = torch.maximum(2 - upper, lower)
    mixed_upper = torch.minimum(2 - lower, upper)
    lower = torch.where(
        all_negative,
        negative_lower,
        torch.where(any_negative, mixed_lower, lower),
    )
    upper = torch.where(
        all_negative,
        negative_upper,
        torch.where(any_negative, mixed_upper, upper),
    )

    mean = clean.sum(dim=0) / counts.clamp(min=1)
    centered = torch.where(finite, inp - mean, torch.zeros_like(inp))
    variance = (centered * centered).sum(dim=0) / counts.clamp(min=1)
    constant = (counts < 2) | (max_abs == 0) | (variance == 0)
    lower = torch.where(constant, torch.ones_like(lower), lower)
    upper = torch.where(constant, torch.ones_like(upper), upper)
    return lower, upper, constant


def _batched_power_lambdas(
    inp: Tensor,
    likelihood: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    left, right, constant = _power_bounds(inp)
    invphi = (math.sqrt(5) - 1) / 2
    c = right - invphi * (right - left)
    d = left + invphi * (right - left)
    fc = likelihood(inp, c)
    fd = likelihood(inp, d)

    for _ in range(YJ_SEARCH_ITERATIONS):
        active = right - left > YJ_SEARCH_ATOL
        move_right = fc < fd
        next_left = torch.where(move_right, c, left)
        next_right = torch.where(move_right, right, d)
        next_c = torch.where(
            move_right,
            d,
            next_right - invphi * (next_right - next_left),
        )
        next_d = torch.where(
            move_right,
            next_left + invphi * (next_right - next_left),
            c,
        )
        candidate = torch.where(move_right, next_d, next_c)
        candidate_score = likelihood(inp, candidate)
        next_fc = torch.where(move_right, fd, candidate_score)
        next_fd = torch.where(move_right, candidate_score, fc)

        left = torch.where(active, next_left, left)
        right = torch.where(active, next_right, right)
        c = torch.where(active, next_c, c)
        d = torch.where(active, next_d, d)
        fc = torch.where(active, next_fc, fc)
        fd = torch.where(active, next_fd, fd)

    lambdas = ((left + right) / 2).to(dtype=inp.dtype)
    return torch.where(constant, torch.ones_like(lambdas), lambdas)


def _nan_mean_var(inp: Tensor) -> tuple[Tensor, Tensor]:
    finite = inp.isfinite()
    counts = finite.sum(dim=0)
    safe_counts = counts.clamp(min=1)
    clean = torch.where(finite, inp, torch.zeros_like(inp))
    mean = clean.sum(dim=0) / safe_counts
    centered = torch.where(finite, inp - mean, torch.zeros_like(inp))
    variance = (centered * centered).sum(dim=0) / safe_counts
    nan = torch.full_like(mean, torch.nan)
    return (
        torch.where(counts > 0, mean, nan),
        torch.where(counts > 0, variance, nan),
    )


def _standard_scale_fit_state(
    inp: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    mean = inp.mean(dim=0)
    if inp.size(0) > 1:
        scale = inp.var(dim=0, correction=0).sqrt()
    else:
        scale = inp.new_zeros(inp.shape[1])
    return mean, scale + epsilon


def _standard_scale_fit_transform(
    inp: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mean, scale = _standard_scale_fit_state(inp, epsilon=epsilon)
    return (inp - mean) / scale, mean, scale


def _standard_scale_inverse(
    inp: Tensor,
    mean: Tensor,
    scale: Tensor,
) -> Tensor:
    return inp * scale + mean


def _sigma_clip_fit_state(
    inp: Tensor,
    *,
    threshold: float,
) -> tuple[Tensor, Tensor]:
    min_std = inp.new_tensor(1e-6)
    mean = torch.nanmean(inp, dim=0)
    std = _nanstd(inp, dim=0)
    std = torch.where(std.isnan(), min_std, std)
    std = torch.maximum(std, min_std)

    inf = inp.new_tensor(float("inf"))
    lower = torch.where(mean.isnan(), -inf, mean - threshold * std)
    upper = torch.where(mean.isnan(), inf, mean + threshold * std)
    clean = torch.where((inp < lower) | (inp > upper), torch.nan, inp)

    mean_clean = torch.nanmean(clean, dim=0)
    std_clean = _nanstd(clean, dim=0)
    mean = torch.where(mean_clean.isnan(), mean, mean_clean)
    std = torch.where(std_clean.isnan(), std, std_clean)
    std = torch.maximum(std, min_std)
    lower = torch.where(mean.isnan(), -inf, mean - threshold * std)
    upper = torch.where(mean.isnan(), inf, mean + threshold * std)
    return lower, upper


def _sigma_clip_transform(
    inp: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> Tensor:
    log_abs = inp.abs().log1p()
    clipped = torch.maximum(-log_abs + lower, inp)
    return torch.minimum(log_abs + upper, clipped)


def _sigma_clip_fit_transform(
    inp: Tensor,
    *,
    threshold: float,
) -> Tensor:
    lower, upper = _sigma_clip_fit_state(inp, threshold=threshold)
    return _sigma_clip_transform(inp, lower, upper)


def _feature_shift_permutation(
    n_features: int,
    offset: int,
    *,
    device: torch.device,
) -> Tensor:
    return (torch.arange(n_features, device=device) + offset) % n_features


def _category_shift_permutation(
    n_classes: int,
    offset: int,
    *,
    device: torch.device,
) -> Tensor:
    return (torch.arange(n_classes, device=device) - offset) % n_classes


def _categorical_observed_mask(
    codes: Tensor,
    *,
    category_count: int,
) -> Tensor:
    valid = codes >= 0
    indices = codes.clamp(min=0, max=category_count - 1).T.to(torch.long)
    observed = torch.zeros(
        codes.shape[1],
        category_count,
        dtype=torch.int64,
        device=codes.device,
    )
    observed.scatter_reduce_(
        dim=1,
        index=indices,
        src=valid.T.to(torch.int64),
        reduce="amax",
        include_self=True,
    )
    return observed.bool()


def _categorical_dense_sorted_mapping(
    codes: Tensor,
    *,
    category_count: int,
) -> Tensor:
    observed = _categorical_observed_mask(
        codes,
        category_count=category_count,
    )
    ranks = observed.to(torch.long).cumsum(dim=1) - 1
    missing = torch.full_like(ranks, -1)
    return torch.where(observed, ranks, missing)


def _categorical_dense_sorted_fit_transform(
    codes: Tensor,
    *,
    category_count: int,
) -> Tensor:
    mapping = _categorical_dense_sorted_mapping(
        codes,
        category_count=category_count,
    )
    return _categorical_lookup_transform(codes, mapping)


def _categorical_lookup_transform(codes: Tensor, mapping: Tensor) -> Tensor:
    in_range = (codes >= 0) & (codes < mapping.shape[1])
    indices = codes.clamp(min=0, max=mapping.shape[1] - 1).T.to(torch.long)
    transformed = mapping.gather(dim=1, index=indices).T.to(codes.dtype)
    missing = torch.full_like(transformed, -1)
    return torch.where(in_range, transformed, missing)


def _category_permutation_transform(
    codes: Tensor,
    permutation: Tensor,
) -> Tensor:
    valid = (codes >= 0) & (codes < permutation.numel())
    indices = codes.clamp(min=0, max=permutation.numel() - 1).to(torch.long)
    transformed = permutation[indices].to(codes.dtype)
    return torch.where(valid, transformed, codes)


def _quantile_fit_preindexed(
    inp: Tensor,
    references: Tensor,
    indices: Tensor,
) -> Tensor:
    return torch.nanquantile(inp[indices], references, dim=0)


def _quantile_fit_transform_preindexed(
    inp: Tensor,
    references: Tensor,
    indices: Tensor,
) -> Tensor:
    quantiles = _quantile_fit_preindexed(inp, references, indices)
    return _vectorized_quantile_row_major(inp, quantiles, references)


def _batched_power_fit(
    inp: Tensor,
    likelihood: Callable[[Tensor, Tensor], Tensor],
    transform: Callable[[Tensor, Tensor], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    lambdas = _batched_power_lambdas(inp, likelihood)
    transformed = transform(inp, lambdas)
    mean, variance = _nan_mean_var(transformed)
    scale = variance.sqrt()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return lambdas, mean, scale


def _batched_power_fit_transform(
    inp: Tensor,
    likelihood: Callable[[Tensor, Tensor], Tensor],
    transform: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    lambdas, mean, scale = _batched_power_fit(inp, likelihood, transform)
    return (transform(inp, lambdas) - mean) / scale


def _power_fit_transform_from_state(
    inp: Tensor,
    state: tuple[Tensor, Tensor, Tensor],
    transform: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    lambdas, mean, scale = state
    return (transform(inp, lambdas) - mean) / scale


def _power_value_derivatives(
    log_value: Tensor,
    parameter: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    scaled = parameter * log_value
    exponential = scaled.exp()
    value = exponential.sub(1) / parameter
    first = (exponential * (scaled - 1) + 1) / parameter.square()
    second = (
        exponential * (scaled.square() - 2 * scaled + 2) - 2
    ) / parameter.pow(3)

    small = parameter.abs() < 1e-3
    square = log_value.square()
    cube = square * log_value
    fourth = cube * log_value
    series_value = (
        log_value
        + parameter * square / 2
        + parameter.square() * cube / 6
        + parameter.pow(3) * fourth / 24
    )
    series_first = square / 2 + parameter * cube / 3
    series_first = series_first + parameter.square() * fourth / 8
    series_second = cube / 3 + parameter * fourth / 4
    return (
        torch.where(small, series_value, value),
        torch.where(small, series_first, first),
        torch.where(small, series_second, second),
    )


def _power_transform_derivatives(
    inp: Tensor,
    lambdas: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    lambdas = lambdas.to(dtype=inp.dtype).unsqueeze(0)
    positive = _power_value_derivatives(
        inp.clamp_min(0).log1p(),
        lambdas,
    )
    negative = _power_value_derivatives(
        (-inp).clamp_min(0).log1p(),
        2 - lambdas,
    )
    mask = inp >= 0
    return (
        torch.where(mask, positive[0], -negative[0]),
        torch.where(mask, positive[1], negative[1]),
        torch.where(mask, positive[2], -negative[2]),
    )


def _newton_power_fit(inp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    lower, upper, constant = _power_bounds(inp)
    lower = lower.to(dtype=inp.dtype)
    upper = upper.to(dtype=inp.dtype)
    finite = inp.isfinite()
    counts = finite.sum(dim=0)
    safe_counts = counts.clamp(min=1)
    clean = torch.where(finite, inp, torch.zeros_like(inp))
    jacobian = (clean.sign() * clean.abs().log1p()).sum(dim=0)
    lambdas = torch.ones(inp.shape[1], dtype=inp.dtype, device=inp.device)

    for _ in range(4):
        value, first, second = _power_transform_derivatives(clean, lambdas)
        value = torch.where(finite, value, torch.zeros_like(value))
        first = torch.where(finite, first, torch.zeros_like(first))
        second = torch.where(finite, second, torch.zeros_like(second))
        mean = value.sum(dim=0) / safe_counts
        first_mean = first.sum(dim=0) / safe_counts
        centered = torch.where(finite, value - mean, torch.zeros_like(value))
        centered_first = torch.where(
            finite,
            first - first_mean,
            torch.zeros_like(first),
        )
        variance = centered.square().sum(dim=0) / safe_counts
        variance_first = 2 * (centered * first).sum(dim=0) / safe_counts
        variance_second = (
            2
            * (centered_first.square() + centered * second).sum(dim=0)
            / safe_counts
        )
        gradient = -counts / 2 * variance_first / variance + jacobian
        hessian = (
            -counts
            / 2
            * (
                variance_second / variance
                - (variance_first / variance).square()
            )
        )
        step = (gradient / hessian).clamp(-1, 1)
        candidate = (lambdas - step).maximum(lower).minimum(upper)
        lambdas = torch.where(
            constant | ~candidate.isfinite(),
            lambdas,
            candidate,
        )

    lambdas = torch.where(constant, torch.ones_like(lambdas), lambdas)
    transformed = _vectorized_yeojohnson(clean, lambdas)
    transformed = torch.where(finite, transformed, torch.nan)
    mean, variance = _nan_mean_var(transformed)
    scale = variance.sqrt()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return lambdas, mean, scale


def _vectorized_yeojohnson_inverse(inp: Tensor, lambdas: Tensor) -> Tensor:
    lambdas = lambdas.to(dtype=inp.dtype).unsqueeze(0)
    eps = torch.finfo(inp.dtype).eps

    positive = inp >= 0
    positive_log = (inp.clamp_min(0) * lambdas + 1).log() / lambdas
    positive_value = positive_log.expm1()
    positive_value = torch.where(
        lambdas.abs() < eps,
        inp.clamp_min(0).expm1(),
        positive_value,
    )

    two_minus_lambda = 2 - lambdas
    negative_input = inp.clamp_max(0)
    negative_log = (-(two_minus_lambda) * negative_input + 1).log()
    negative_value = -(negative_log / two_minus_lambda).expm1()
    negative_value = torch.where(
        two_minus_lambda.abs() < eps,
        -(-negative_input).expm1(),
        negative_value,
    )
    return torch.where(positive, positive_value, negative_value)


def _vectorized_power_inverse(
    inp: Tensor,
    lambdas: Tensor,
    mean: Tensor,
    scale: Tensor,
    upper_bound: Tensor,
    maximum: Tensor,
) -> Tensor:
    unscaled = inp * scale + mean
    inverse = _vectorized_yeojohnson_inverse(unscaled, lambdas)
    eps = torch.finfo(inp.dtype).eps
    bounded = torch.minimum(unscaled, upper_bound - eps)
    repaired = _vectorized_yeojohnson_inverse(bounded, lambdas)
    inverse = torch.where(inverse.isinf(), repaired, inverse)
    return torch.where(
        inverse.isinf(),
        torch.fmin(inverse, maximum),
        inverse,
    )


def _batched_inverse_quantile_row_major(
    inp: Tensor,
    quantiles: Tensor,
    references: Tensor,
    *,
    normal: bool,
) -> Tensor:
    values = inp.T.contiguous()
    if normal:
        values = torch.special.ndtr(values)

    quantile_columns = quantiles.T.contiguous()
    n = references.numel()
    if n == 1:
        output = quantile_columns[:, :1].expand_as(values)
    else:
        indices = torch.searchsorted(references, values, right=True)
        indices = indices.clamp(1, n - 1)
        x0 = references[indices - 1]
        x1 = references[indices]
        y0 = quantile_columns.gather(1, indices - 1)
        y1 = quantile_columns.gather(1, indices)
        denominator = x1 - x0
        weight = torch.where(
            denominator != 0,
            (values - x0) / denominator,
            torch.zeros_like(values),
        )
        output = torch.lerp(y0, y1, weight)
        output = torch.where(
            values <= references[0], quantile_columns[:, :1], output
        )
        output = torch.where(
            values >= references[-1], quantile_columns[:, -1:], output
        )

    zero = values.new_zeros(())
    one = values.new_ones(())
    if normal:
        bounds_thresh = values.new_tensor(1e-7)
        lower = values - bounds_thresh < zero
        upper = values + bounds_thresh > one
    else:
        lower = values == zero
        upper = values == one
    output = torch.where(upper, quantile_columns[:, -1:], output)
    output = torch.where(lower, quantile_columns[:, :1], output)
    output = torch.where(values.isfinite(), output, values)
    return output.T.contiguous()


def _batched_interp(
    values: Tensor,
    boundaries: Tensor,
    references: Tensor,
) -> Tensor:
    n = boundaries.shape[-1]
    if n == 1:
        return references[0].expand_as(values)

    indices = torch.searchsorted(boundaries, values, right=True)
    indices = indices.clamp(1, n - 1)
    x0 = boundaries.gather(1, indices - 1)
    x1 = boundaries.gather(1, indices)
    y0 = references[indices - 1]
    y1 = references[indices]
    denominator = x1 - x0
    weight = torch.where(
        denominator != 0,
        (values - x0) / denominator,
        torch.zeros_like(values),
    )
    result = torch.lerp(y0, y1, weight)
    result = torch.where(values <= boundaries[:, :1], references[0], result)
    return torch.where(values >= boundaries[:, -1:], references[-1], result)


def _vectorized_quantile_col_major(
    inp: Tensor,
    quantiles: Tensor,
    references: Tensor,
) -> Tensor:
    forward = _batched_interp(inp, quantiles, references)
    backward = _batched_interp(
        -inp,
        -quantiles.flip(1),
        -references.flip(0),
    )
    transformed = 0.5 * (forward - backward)

    threshold = inp.new_tensor(1e-7)
    lower = inp - threshold < quantiles[:, :1]
    upper = inp + threshold > quantiles[:, -1:]
    transformed = torch.where(upper, torch.ones_like(transformed), transformed)
    transformed = torch.where(
        lower, torch.zeros_like(transformed), transformed
    )
    transformed = torch.special.ndtri(transformed)
    eps = inp.new_tensor(1e-7 - torch.finfo(torch.float64).eps)
    transformed = transformed.clamp(
        torch.special.ndtri(eps),
        torch.special.ndtri(1 - eps),
    )
    return transformed.masked_fill(inp.isnan(), torch.nan)


def _vectorized_quantile_row_major(
    inp: Tensor,
    quantiles: Tensor,
    references: Tensor,
) -> Tensor:
    column_major = inp.T.contiguous()
    transformed = _vectorized_quantile_col_major(
        column_major,
        quantiles.T.contiguous(),
        references,
    )
    return transformed.T.contiguous()


def _feature_batched_quantile_row_major(
    inp: Tensor,
    quantiles: Tensor,
    references: Tensor,
    *,
    batch_size: int = 32,
) -> Tensor:
    outputs = [
        _vectorized_quantile_row_major(
            inp[:, start : start + batch_size],
            quantiles[:, start : start + batch_size],
            references,
        )
        for start in range(0, inp.shape[1], batch_size)
    ]
    return torch.cat(outputs, dim=1)


def _mixed_precision_quantile_row_major(
    inp: Tensor,
    quantiles: Tensor,
    references: Tensor,
) -> Tensor:
    column_major = inp.T.contiguous()
    column_quantiles = quantiles.T.contiguous()
    forward = _batched_interp(
        column_major,
        column_quantiles,
        references,
    )
    backward = _batched_interp(
        -column_major,
        -column_quantiles.flip(1),
        -references.flip(0),
    )
    transformed = 0.5 * (forward - backward)
    threshold = inp.new_tensor(1e-7)
    lower = column_major - threshold < column_quantiles[:, :1]
    upper = column_major + threshold > column_quantiles[:, -1:]
    transformed = torch.where(upper, torch.ones_like(transformed), transformed)
    transformed = torch.where(
        lower, torch.zeros_like(transformed), transformed
    )
    transformed = transformed.float()
    eps = transformed.new_tensor(1e-7 - torch.finfo(torch.float64).eps)
    transformed = torch.special.ndtri(transformed).clamp(
        torch.special.ndtri(eps),
        torch.special.ndtri(1 - eps),
    )
    transformed = transformed.masked_fill(column_major.isnan(), torch.nan)
    return transformed.T.contiguous()


def _max_abs_error(actual: Tensor, expected: Tensor) -> float:
    finite = actual.isfinite() & expected.isfinite()
    if not finite.any():
        return 0.0
    return float((actual[finite] - expected[finite]).abs().max())


def _result(
    *,
    processor: str,
    operation: str,
    candidate: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    characteristics: str,
    timing: Timing,
    error: float | None,
    correct: bool,
    repetitions: int,
    warmups: int,
    device: torch.device,
) -> Result:
    return Result(
        processor=processor,
        operation=operation,
        candidate=candidate,
        dtype=str(dtype),
        input_shape=shape,
        dataset_characteristics=characteristics,
        median_ms=timing.median_ms,
        p95_ms=timing.p95_ms,
        cuda_median_ms=timing.cuda_median_ms,
        cuda_p95_ms=timing.cuda_p95_ms,
        host_overhead_median_ms=timing.host_overhead_median_ms,
        peak_memory_bytes=timing.peak_memory_bytes,
        throughput_values_per_second=(
            math.prod(shape) / (timing.median_ms / 1000)
        ),
        max_abs_error=error,
        correctness_status="pass" if correct else "fail",
        repetitions=repetitions,
        warmups=warmups,
        gpu_model=torch.cuda.get_device_name(device),
    )


def _append_transfer_results(
    results: list[Result],
    *,
    processor: str,
    operation: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    characteristics: str,
    repetitions: int,
    warmups: int,
    device: torch.device,
) -> None:
    host_input = torch.empty(input_shape, dtype=torch.float32, pin_memory=True)
    gpu_input = torch.empty(input_shape, dtype=torch.float32, device=device)
    host_output = torch.empty(
        output_shape,
        dtype=torch.float32,
        pin_memory=True,
    )
    gpu_output = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=device,
    )
    for direction, shape, transfer in (
        (
            "host_to_device",
            input_shape,
            lambda: gpu_input.copy_(host_input, non_blocking=True),
        ),
        (
            "device_to_host",
            output_shape,
            lambda: host_output.copy_(gpu_output, non_blocking=True),
        ),
    ):
        timing = _measure_cuda(
            transfer,
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        results.append(
            _result(
                processor=processor,
                operation=f"{operation}_{direction}",
                candidate="pinned_preallocated_transfer",
                dtype=torch.float32,
                shape=shape,
                characteristics=characteristics,
                timing=timing,
                error=None,
                correct=True,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )


def _idle_synchronize_timing(device: torch.device) -> tuple[float, float]:
    durations: list[float] = []
    for _ in range(1_000):
        started = time.perf_counter_ns()
        torch.cuda.synchronize(device)
        durations.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(durations), _p95(durations)


def _append_other_processor_results(
    results: list[Result],
    *,
    workload,
    characteristics: str,
    repetitions: int,
    warmups: int,
    device: torch.device,
) -> None:
    numeric = workload.x.select_stypes(Stype.numerical)
    fit_table = numeric[: workload.train_rows]
    fit_input = fit_table.numerical
    transform_input = numeric.numerical

    clip = Clip(min_value=-100.0, max_value=100.0)
    clip_expected = clip.transform(numeric).numerical
    timing = _measure_cuda(
        lambda: clip.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Clip",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: transform_input.clamp(min=-100.0, max=100.0),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    clip_direct = transform_input.clamp(min=-100.0, max=100.0)
    error = _max_abs_error(clip_direct, clip_expected)
    results.append(
        _result(
            processor="Clip",
            operation="transform",
            candidate="direct_clamp",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                clip_direct,
                clip_expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    standard_expected = (
        StandardScale(epsilon=1e-6).fit_transform(fit_table).numerical
    )
    timing = _measure_cuda(
        lambda: StandardScale(epsilon=1e-6).fit_transform(fit_table).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="StandardScale",
            operation="fit_transform",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_fit_transform(fit_input, epsilon=1e-6)[0],
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    standard_direct = _standard_scale_fit_transform(fit_input, epsilon=1e-6)[0]
    error = _max_abs_error(standard_direct, standard_expected)
    results.append(
        _result(
            processor="StandardScale",
            operation="fit_transform",
            candidate="direct_reduce_affine",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                standard_direct,
                standard_expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    sigma_expected = (
        SigmaClip(threshold=4.0).fit_transform(fit_table).numerical
    )
    timing = _measure_cuda(
        lambda: SigmaClip(threshold=4.0).fit_transform(fit_table).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="SigmaClip",
            operation="fit_transform",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _sigma_clip_fit_transform(fit_input, threshold=4.0),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    sigma_direct = _sigma_clip_fit_transform(fit_input, threshold=4.0)
    error = _max_abs_error(sigma_direct, sigma_expected)
    results.append(
        _result(
            processor="SigmaClip",
            operation="fit_transform",
            candidate="direct_two_pass_soft_clip",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                sigma_direct,
                sigma_expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    references = torch.linspace(
        0,
        1,
        1000,
        device=fit_input.device,
        dtype=fit_input.dtype,
    )
    torch.manual_seed(0)
    quantile_expected = Quantile(output_distribution="normal").fit(fit_table)
    torch.manual_seed(0)
    quantile_indices = torch.randperm(
        fit_input.shape[0],
        device=device,
    )[:10_000]
    direct_quantiles = _quantile_fit_preindexed(
        fit_input,
        references,
        quantile_indices,
    )
    timing = _measure_cuda(
        lambda: Quantile(output_distribution="normal").fit(fit_table),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Quantile",
            operation="fit",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _quantile_fit_preindexed(
            fit_input,
            references,
            quantile_indices,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    error = _max_abs_error(direct_quantiles, quantile_expected.quantiles)
    results.append(
        _result(
            processor="Quantile",
            operation="fit",
            candidate="direct_nanquantile_preindexed",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                direct_quantiles,
                quantile_expected.quantiles,
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    categorical = workload.x.select_stypes(Stype.categorical)
    if categorical.size(-1) > 0:
        categorical_fit = categorical[: workload.train_rows]
        fit_codes = categorical_fit.categorical.as_tensor()
        transform_codes = categorical.categorical.as_tensor()
        align = CategoricalAlign(order="sorted").fit(categorical_fit)
        category_count = max(
            category.numel() for category in categorical.categorical.categories
        )
        expected_mask = torch.zeros(
            fit_codes.shape[1],
            category_count,
            dtype=torch.bool,
            device=device,
        )
        for index, category in enumerate(align._categories):
            expected_mask[index, category.to(device=device).to(torch.long)] = (
                True
            )

        timing = _measure_cuda(
            lambda: CategoricalAlign(order="sorted").fit(categorical_fit),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="fit",
                candidate="current",
                dtype=fit_codes.dtype,
                shape=tuple(fit_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=0.0,
                correct=True,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )
        timing = _measure_cuda(
            lambda: _categorical_observed_mask(
                fit_codes,
                category_count=category_count,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        observed_mask = _categorical_observed_mask(
            fit_codes,
            category_count=category_count,
        )
        mask_correct = torch.equal(observed_mask, expected_mask)
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="fit",
                candidate="dense_observed_mask",
                dtype=fit_codes.dtype,
                shape=tuple(fit_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=0.0 if mask_correct else 1.0,
                correct=mask_correct,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

        aligned_expected = align.transform(categorical).categorical.as_tensor()
        mappings = torch.stack(
            [
                CategoricalAlign._category_mapping(
                    actual=actual,
                    expected=expected,
                    device=device,
                    column=f"cat_{index}",
                )
                for index, (actual, expected) in enumerate(
                    zip(
                        categorical.categorical.categories,
                        align._categories,
                        strict=True,
                    )
                )
            ]
        )
        timing = _measure_cuda(
            lambda: align.transform(categorical).categorical.as_tensor(),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="transform",
                candidate="current",
                dtype=transform_codes.dtype,
                shape=tuple(transform_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=0.0,
                correct=True,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )
        timing = _measure_cuda(
            lambda: _categorical_lookup_transform(transform_codes, mappings),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        aligned_direct = _categorical_lookup_transform(
            transform_codes, mappings
        )
        align_error = _max_abs_error(
            aligned_direct.float(),
            aligned_expected.float(),
        )
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="transform",
                candidate="direct_lookup_gather",
                dtype=transform_codes.dtype,
                shape=tuple(transform_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=align_error,
                correct=torch.equal(aligned_direct, aligned_expected),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    target_codes = workload.y.categorical.as_tensor()
    torch.manual_seed(0)
    target_shuffle = CategoryShuffle(method="shift").fit(workload.y)
    target_expected = target_shuffle.transform(
        workload.y
    ).categorical.as_tensor()
    permutation = target_shuffle.permutations
    timing = _measure_cuda(
        lambda: (
            CategoryShuffle(method="shift")
            .fit_transform(workload.y)
            .categorical.as_tensor()
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="fit_transform",
            candidate="current",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _category_permutation_transform(target_codes, permutation),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    target_direct = _category_permutation_transform(target_codes, permutation)
    target_error = _max_abs_error(
        target_direct.float(), target_expected.float()
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="fit_transform",
            candidate="direct_permutation_gather",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=target_error,
            correct=torch.equal(target_direct, target_expected),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    regression = build_workload(
        size="large",
        task="regression",
        characteristics=workload.characteristics,
        device=device,
    )
    regression_target = StandardScale().fit(regression.y)
    inverse_input = regression_target.transform(regression.y)
    inverse_tensor = inverse_input.numerical
    inverse_expected = regression_target.inverse_transform(
        inverse_input
    ).numerical
    timing = _measure_cuda(
        lambda: regression_target.inverse_transform(inverse_input).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="inverse_transform",
            candidate="current",
            dtype=inverse_tensor.dtype,
            shape=tuple(inverse_tensor.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_inverse(
            inverse_tensor,
            regression_target.mean,
            regression_target.scale,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    inverse_direct = _standard_scale_inverse(
        inverse_tensor,
        regression_target.mean,
        regression_target.scale,
    )
    inverse_error = _max_abs_error(inverse_direct, inverse_expected)
    results.append(
        _result(
            processor="RegressionTarget",
            operation="inverse_transform",
            candidate="direct_affine",
            dtype=inverse_tensor.dtype,
            shape=tuple(inverse_tensor.shape),
            characteristics=characteristics,
            timing=timing,
            error=inverse_error,
            correct=torch.equal(inverse_direct, inverse_expected),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    scale_processor = StandardScale(epsilon=1e-6).fit(fit_table)
    scale_mean, scale_scale = _standard_scale_fit_state(
        fit_input,
        epsilon=1e-6,
    )
    timing = _measure_cuda(
        lambda: StandardScale(epsilon=1e-6).fit(fit_table),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="StandardScale",
            operation="fit",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_fit_state(fit_input, epsilon=1e-6),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    state_error = max(
        _max_abs_error(scale_mean, scale_processor.mean),
        _max_abs_error(scale_scale, scale_processor.scale),
    )
    results.append(
        _result(
            processor="StandardScale",
            operation="fit",
            candidate="direct_reduce_state",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=state_error,
            correct=(
                torch.allclose(
                    scale_mean,
                    scale_processor.mean,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
                and torch.allclose(
                    scale_scale,
                    scale_processor.scale,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    scale_transform_expected = scale_processor.transform(numeric).numerical
    timing = _measure_cuda(
        lambda: scale_processor.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="StandardScale",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: (
            (transform_input - scale_processor.mean) / scale_processor.scale
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    scale_transform_direct = (
        transform_input - scale_processor.mean
    ) / scale_processor.scale
    error = _max_abs_error(scale_transform_direct, scale_transform_expected)
    results.append(
        _result(
            processor="StandardScale",
            operation="transform",
            candidate="direct_affine",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                scale_transform_direct,
                scale_transform_expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    scale_inverse_expected = scale_processor.inverse_transform(
        numeric
    ).numerical
    timing = _measure_cuda(
        lambda: scale_processor.inverse_transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="StandardScale",
            operation="inverse_transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_inverse(
            transform_input,
            scale_processor.mean,
            scale_processor.scale,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    scale_inverse_direct = _standard_scale_inverse(
        transform_input,
        scale_processor.mean,
        scale_processor.scale,
    )
    error = _max_abs_error(scale_inverse_direct, scale_inverse_expected)
    results.append(
        _result(
            processor="StandardScale",
            operation="inverse_transform",
            candidate="direct_affine",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                scale_inverse_direct,
                scale_inverse_expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    sigma_processor = SigmaClip(threshold=4.0).fit(fit_table)
    sigma_lower, sigma_upper = _sigma_clip_fit_state(
        fit_input,
        threshold=4.0,
    )
    timing = _measure_cuda(
        lambda: SigmaClip(threshold=4.0).fit(fit_table),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="SigmaClip",
            operation="fit",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _sigma_clip_fit_state(fit_input, threshold=4.0),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    state_error = max(
        _max_abs_error(sigma_lower, sigma_processor.lower_bound),
        _max_abs_error(sigma_upper, sigma_processor.upper_bound),
    )
    results.append(
        _result(
            processor="SigmaClip",
            operation="fit",
            candidate="direct_two_pass_bounds",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=state_error,
            correct=(
                torch.allclose(
                    sigma_lower,
                    sigma_processor.lower_bound,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
                and torch.allclose(
                    sigma_upper,
                    sigma_processor.upper_bound,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=True,
                )
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    sigma_transform_expected = sigma_processor.transform(numeric).numerical
    timing = _measure_cuda(
        lambda: sigma_processor.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="SigmaClip",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _sigma_clip_transform(
            transform_input,
            sigma_processor.lower_bound,
            sigma_processor.upper_bound,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    sigma_transform_direct = _sigma_clip_transform(
        transform_input,
        sigma_processor.lower_bound,
        sigma_processor.upper_bound,
    )
    error = _max_abs_error(sigma_transform_direct, sigma_transform_expected)
    results.append(
        _result(
            processor="SigmaClip",
            operation="transform",
            candidate="direct_soft_clip",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                sigma_transform_direct,
                sigma_transform_expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    torch.manual_seed(0)
    feature_permute = FeaturePermute(method="shift").fit(fit_table)
    feature_offset = int(feature_permute.permutation[0].item())
    feature_direct_permutation = _feature_shift_permutation(
        fit_input.shape[1],
        feature_offset,
        device=device,
    )
    timing = _measure_cuda(
        lambda: FeaturePermute(method="shift").fit(fit_table),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="fit",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _feature_shift_permutation(
            fit_input.shape[1],
            feature_offset,
            device=device,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    permutation_error = _max_abs_error(
        feature_direct_permutation.float(),
        feature_permute.permutation.float(),
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="fit",
            candidate="direct_shift_permutation",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=permutation_error,
            correct=torch.equal(
                feature_direct_permutation,
                feature_permute.permutation,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    feature_fit_transform_expected = feature_permute.transform(
        fit_table
    ).numerical
    timing = _measure_cuda(
        lambda: (
            FeaturePermute(method="shift").fit_transform(fit_table).numerical
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="fit_transform",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: fit_input.index_select(-1, feature_direct_permutation),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    feature_fit_transform_direct = fit_input.index_select(
        -1,
        feature_direct_permutation,
    )
    error = _max_abs_error(
        feature_fit_transform_direct,
        feature_fit_transform_expected,
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="fit_transform",
            candidate="direct_index_select",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                feature_fit_transform_direct,
                feature_fit_transform_expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    feature_transform_expected = feature_permute.transform(numeric).numerical
    timing = _measure_cuda(
        lambda: feature_permute.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: transform_input.index_select(-1, feature_permute.permutation),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    feature_transform_direct = transform_input.index_select(
        -1,
        feature_permute.permutation,
    )
    error = _max_abs_error(
        feature_transform_direct, feature_transform_expected
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="transform",
            candidate="direct_index_select",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                feature_transform_direct,
                feature_transform_expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    feature_inverse_input = feature_permute.transform(numeric)
    feature_inverse_expected = feature_permute.inverse_transform(
        feature_inverse_input
    ).numerical
    inverse_permutation = feature_permute.permutation.argsort()
    timing = _measure_cuda(
        lambda: (
            feature_permute.inverse_transform(feature_inverse_input).numerical
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="FeaturePermute",
            operation="inverse_transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: feature_inverse_input.numerical.index_select(
            -1,
            inverse_permutation,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    feature_inverse_direct = feature_inverse_input.numerical.index_select(
        -1,
        inverse_permutation,
    )
    error = _max_abs_error(feature_inverse_direct, feature_inverse_expected)
    results.append(
        _result(
            processor="FeaturePermute",
            operation="inverse_transform",
            candidate="direct_index_select",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                feature_inverse_direct,
                feature_inverse_expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    logits = torch.zeros(
        workload.rows,
        10,
        dtype=transform_input.dtype,
        device=device,
    )
    logits_table = TableTensor(
        columns={"numerical": tuple(f"logit_{i}" for i in range(10))},
        numerical=logits,
    )
    softmax = SoftmaxTemperature(temperature=0.9)
    softmax_expected = softmax.transform(logits_table).numerical
    timing = _measure_cuda(
        lambda: softmax.transform(logits_table).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="SoftmaxTemperature",
            operation="transform",
            candidate="current",
            dtype=logits.dtype,
            shape=tuple(logits.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: torch.softmax(logits / 0.9, dim=-1),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    softmax_direct = torch.softmax(logits / 0.9, dim=-1)
    error = _max_abs_error(softmax_direct, softmax_expected)
    results.append(
        _result(
            processor="SoftmaxTemperature",
            operation="transform",
            candidate="direct_softmax",
            dtype=logits.dtype,
            shape=tuple(logits.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(softmax_direct, softmax_expected),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    categorical = workload.x.select_stypes(Stype.categorical)
    if categorical.size(-1) > 0:
        categorical_fit = categorical[: workload.train_rows]
        fit_codes = categorical_fit.categorical.as_tensor()
        category_count = max(
            category.numel() for category in categorical.categorical.categories
        )
        align = CategoricalAlign(order="sorted").fit(categorical_fit)
        fit_aligned_expected = align.transform(
            categorical_fit
        ).categorical.as_tensor()
        timing = _measure_cuda(
            lambda: (
                CategoricalAlign(order="sorted")
                .fit_transform(categorical_fit)
                .categorical.as_tensor()
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="fit_transform",
                candidate="current",
                dtype=fit_codes.dtype,
                shape=tuple(fit_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=0.0,
                correct=True,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )
        timing = _measure_cuda(
            lambda: _categorical_dense_sorted_fit_transform(
                fit_codes,
                category_count=category_count,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        fit_aligned_direct = _categorical_dense_sorted_fit_transform(
            fit_codes,
            category_count=category_count,
        )
        error = _max_abs_error(
            fit_aligned_direct.float(),
            fit_aligned_expected.float(),
        )
        results.append(
            _result(
                processor="CategoricalAlign",
                operation="fit_transform",
                candidate="dense_observed_lookup",
                dtype=fit_codes.dtype,
                shape=tuple(fit_codes.shape),
                characteristics=characteristics,
                timing=timing,
                error=error,
                correct=torch.equal(fit_aligned_direct, fit_aligned_expected),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    target_codes = workload.y.categorical.as_tensor()
    torch.manual_seed(0)
    target_shuffle = CategoryShuffle(method="shift").fit(workload.y)
    target_offset = int((-target_shuffle.permutations[0]).remainder(2).item())
    timing = _measure_cuda(
        lambda: CategoryShuffle(method="shift").fit(workload.y),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="fit",
            candidate="current",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _category_shift_permutation(
            2,
            target_offset,
            device=device,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    target_fit_direct = _category_shift_permutation(
        2,
        target_offset,
        device=device,
    )
    error = _max_abs_error(
        target_fit_direct.float(),
        target_shuffle.permutations.float(),
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="fit",
            candidate="direct_shift_permutation",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.equal(
                target_fit_direct, target_shuffle.permutations
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    target_transform_expected = target_shuffle.transform(
        workload.y
    ).categorical.as_tensor()
    timing = _measure_cuda(
        lambda: target_shuffle.transform(workload.y).categorical.as_tensor(),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="transform",
            candidate="current",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _category_permutation_transform(
            target_codes,
            target_shuffle.permutations,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    target_transform_direct = _category_permutation_transform(
        target_codes,
        target_shuffle.permutations,
    )
    error = _max_abs_error(
        target_transform_direct.float(),
        target_transform_expected.float(),
    )
    results.append(
        _result(
            processor="ClassificationTarget",
            operation="transform",
            candidate="direct_permutation_gather",
            dtype=target_codes.dtype,
            shape=tuple(target_codes.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.equal(
                target_transform_direct, target_transform_expected
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    regression = build_workload(
        size="large",
        task="regression",
        characteristics=workload.characteristics,
        device=device,
    )
    regression_input = regression.y.numerical
    regression_target = StandardScale().fit(regression.y)
    regression_mean, regression_scale = _standard_scale_fit_state(
        regression_input,
        epsilon=0.0,
    )
    timing = _measure_cuda(
        lambda: StandardScale().fit(regression.y),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="fit",
            candidate="current",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_fit_state(regression_input, epsilon=0.0),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    state_error = max(
        _max_abs_error(regression_mean, regression_target.mean),
        _max_abs_error(regression_scale, regression_target.scale),
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="fit",
            candidate="direct_reduce_state",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=state_error,
            correct=(
                torch.allclose(regression_mean, regression_target.mean)
                and torch.allclose(regression_scale, regression_target.scale)
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    regression_fit_transform_expected = regression_target.transform(
        regression.y
    ).numerical
    timing = _measure_cuda(
        lambda: StandardScale().fit_transform(regression.y).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="fit_transform",
            candidate="current",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _standard_scale_fit_transform(
            regression_input,
            epsilon=0.0,
        )[0],
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    regression_fit_transform_direct = _standard_scale_fit_transform(
        regression_input,
        epsilon=0.0,
    )[0]
    error = _max_abs_error(
        regression_fit_transform_direct,
        regression_fit_transform_expected,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="fit_transform",
            candidate="direct_reduce_affine",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                regression_fit_transform_direct,
                regression_fit_transform_expected,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    regression_transform_expected = regression_target.transform(
        regression.y
    ).numerical
    timing = _measure_cuda(
        lambda: regression_target.transform(regression.y).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="transform",
            candidate="current",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: (
            (regression_input - regression_target.mean)
            / regression_target.scale
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    regression_transform_direct = (
        regression_input - regression_target.mean
    ) / regression_target.scale
    error = _max_abs_error(
        regression_transform_direct,
        regression_transform_expected,
    )
    results.append(
        _result(
            processor="RegressionTarget",
            operation="transform",
            candidate="direct_affine",
            dtype=regression_input.dtype,
            shape=tuple(regression_input.shape),
            characteristics=characteristics,
            timing=timing,
            error=error,
            correct=torch.allclose(
                regression_transform_direct,
                regression_transform_expected,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )


def run(
    *,
    repetitions: int,
    warmups: int,
    compile_candidates: bool,
) -> list[Result]:
    """Run the representative L4 speed-of-light investigation."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")
    characteristics = Characteristics(
        constant=True,
        hard_outlier=True,
        sigma_outlier=True,
        categorical=True,
        missing=True,
        unknown_category=True,
        high_cardinality=True,
    )
    workload = build_workload(
        size="large",
        task="classification",
        characteristics=characteristics,
        device=device,
    )
    numeric = workload.x.select_stypes(Stype.numerical)
    fit_table = numeric[: workload.train_rows]
    fit_input = fit_table.numerical
    transform_input = numeric.numerical
    label = characteristics.label
    results: list[Result] = []
    _append_other_processor_results(
        results,
        workload=workload,
        characteristics=label,
        repetitions=repetitions,
        warmups=warmups,
        device=device,
    )

    current_power = Power().fit(fit_table)
    current_power_output = current_power.transform(fit_table).numerical
    current_power_lambdas = current_power.lambdas.clone()

    timing = _measure_cuda(
        lambda: Power().fit(fit_table),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Power",
            operation="fit",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _batched_power_fit(
            fit_input,
            _power_log_likelihood,
            _vectorized_yeojohnson,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    candidate_lambdas, candidate_mean, candidate_scale = _batched_power_fit(
        fit_input,
        _power_log_likelihood,
        _vectorized_yeojohnson,
    )
    candidate_output = (
        _vectorized_yeojohnson(fit_input, candidate_lambdas) - candidate_mean
    ) / candidate_scale
    lambda_error = _max_abs_error(candidate_lambdas, current_power_lambdas)
    output_error = _max_abs_error(candidate_output, current_power_output)
    results.append(
        _result(
            processor="Power",
            operation="fit",
            candidate="batched_golden_eager",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=max(lambda_error, output_error),
            correct=(
                lambda_error <= POWER_LAMBDA_ATOL
                and output_error <= POWER_OUTPUT_ATOL
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _newton_power_fit(fit_input),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    newton_state = _newton_power_fit(fit_input)
    newton_output = (
        _vectorized_yeojohnson(fit_input, newton_state[0]) - newton_state[1]
    ) / newton_state[2]
    lambda_error = _max_abs_error(newton_state[0], current_power_lambdas)
    output_error = _max_abs_error(newton_output, current_power_output)
    results.append(
        _result(
            processor="Power",
            operation="fit",
            candidate="batched_newton_eager",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=max(lambda_error, output_error),
            correct=(
                lambda_error <= POWER_LAMBDA_ATOL
                and output_error <= POWER_OUTPUT_ATOL
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    compiled_likelihood = _power_log_likelihood
    compiled_transform = _vectorized_yeojohnson
    if compile_candidates:
        compiled_newton_fit = torch.compile(
            _newton_power_fit,
            fullgraph=True,
        )
        compiled_newton_fit(fit_input)
        torch.cuda.synchronize(device)
        compiled_likelihood = torch.compile(
            _power_log_likelihood,
            fullgraph=True,
        )
        compiled_transform = torch.compile(
            _vectorized_yeojohnson,
            fullgraph=True,
        )

        def compiled_transform_preserve_nan(
            inp: Tensor,
            lambdas: Tensor,
        ) -> Tensor:
            output = compiled_transform(inp, lambdas)
            return output.masked_fill(inp.isnan(), torch.nan)

        _batched_power_fit(
            fit_input,
            compiled_likelihood,
            compiled_transform_preserve_nan,
        )
        torch.cuda.synchronize(device)
        timing = _measure_cuda(
            lambda: _batched_power_fit(
                fit_input,
                compiled_likelihood,
                compiled_transform_preserve_nan,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_state = _batched_power_fit(
            fit_input,
            compiled_likelihood,
            compiled_transform_preserve_nan,
        )
        compiled_output = (
            compiled_transform_preserve_nan(fit_input, compiled_state[0])
            - compiled_state[1]
        ) / compiled_state[2]
        lambda_error = _max_abs_error(compiled_state[0], current_power_lambdas)
        output_error = _max_abs_error(compiled_output, current_power_output)
        results.append(
            _result(
                processor="Power",
                operation="fit",
                candidate="batched_golden_compiled",
                dtype=fit_input.dtype,
                shape=tuple(fit_input.shape),
                characteristics=label,
                timing=timing,
                error=max(lambda_error, output_error),
                correct=(
                    lambda_error <= POWER_LAMBDA_ATOL
                    and output_error <= POWER_OUTPUT_ATOL
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

        timing = _measure_cuda(
            lambda: compiled_newton_fit(fit_input),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_newton_state = compiled_newton_fit(fit_input)
        compiled_newton_output = (
            compiled_transform_preserve_nan(
                fit_input,
                compiled_newton_state[0],
            )
            - compiled_newton_state[1]
        ) / compiled_newton_state[2]
        lambda_error = _max_abs_error(
            compiled_newton_state[0],
            current_power_lambdas,
        )
        output_error = _max_abs_error(
            compiled_newton_output,
            current_power_output,
        )
        results.append(
            _result(
                processor="Power",
                operation="fit",
                candidate="batched_newton_compiled",
                dtype=fit_input.dtype,
                shape=tuple(fit_input.shape),
                characteristics=label,
                timing=timing,
                error=max(lambda_error, output_error),
                correct=(
                    lambda_error <= POWER_LAMBDA_ATOL
                    and output_error <= POWER_OUTPUT_ATOL
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    timing = _measure_cuda(
        lambda: Power().fit_transform(fit_table).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Power",
            operation="fit_transform",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _batched_power_fit_transform(
            fit_input,
            _power_log_likelihood,
            _vectorized_yeojohnson,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    fit_transform_output = _batched_power_fit_transform(
        fit_input,
        _power_log_likelihood,
        _vectorized_yeojohnson,
    )
    output_error = _max_abs_error(fit_transform_output, current_power_output)
    results.append(
        _result(
            processor="Power",
            operation="fit_transform",
            candidate="batched_golden_eager",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=output_error,
            correct=output_error <= POWER_OUTPUT_ATOL,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _power_fit_transform_from_state(
            fit_input,
            _newton_power_fit(fit_input),
            _vectorized_yeojohnson,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    newton_fit_transform = (
        _vectorized_yeojohnson(fit_input, newton_state[0]) - newton_state[1]
    ) / newton_state[2]
    output_error = _max_abs_error(newton_fit_transform, current_power_output)
    results.append(
        _result(
            processor="Power",
            operation="fit_transform",
            candidate="batched_newton_eager",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=output_error,
            correct=output_error <= POWER_OUTPUT_ATOL,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    if compile_candidates:
        timing = _measure_cuda(
            lambda: _batched_power_fit_transform(
                fit_input,
                compiled_likelihood,
                compiled_transform_preserve_nan,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_fit_transform = _batched_power_fit_transform(
            fit_input,
            compiled_likelihood,
            compiled_transform_preserve_nan,
        )
        output_error = _max_abs_error(
            compiled_fit_transform,
            current_power_output,
        )
        results.append(
            _result(
                processor="Power",
                operation="fit_transform",
                candidate="batched_golden_compiled",
                dtype=fit_input.dtype,
                shape=tuple(fit_input.shape),
                characteristics=label,
                timing=timing,
                error=output_error,
                correct=output_error <= POWER_OUTPUT_ATOL,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

        timing = _measure_cuda(
            lambda: _power_fit_transform_from_state(
                fit_input,
                compiled_newton_fit(fit_input),
                compiled_transform_preserve_nan,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_newton_fit_transform = (
            compiled_transform_preserve_nan(
                fit_input,
                compiled_newton_state[0],
            )
            - compiled_newton_state[1]
        ) / compiled_newton_state[2]
        output_error = _max_abs_error(
            compiled_newton_fit_transform,
            current_power_output,
        )
        results.append(
            _result(
                processor="Power",
                operation="fit_transform",
                candidate="batched_newton_compiled",
                dtype=fit_input.dtype,
                shape=tuple(fit_input.shape),
                characteristics=label,
                timing=timing,
                error=output_error,
                correct=output_error <= POWER_OUTPUT_ATOL,
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    timing = _measure_cuda(
        lambda: current_power.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    current_transform = current_power.transform(numeric).numerical
    results.append(
        _result(
            processor="Power",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: (
            (
                _vectorized_yeojohnson(transform_input, current_power.lambdas)
                - current_power.mean
            )
            / current_power.scale
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    vectorized_transform = (
        _vectorized_yeojohnson(transform_input, current_power.lambdas)
        - current_power.mean
    ) / current_power.scale
    error = _max_abs_error(vectorized_transform, current_transform)
    results.append(
        _result(
            processor="Power",
            operation="transform",
            candidate="vectorized_eager",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=error <= POWER_OUTPUT_ATOL,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    if compile_candidates:
        timing = _measure_cuda(
            lambda: (
                (
                    compiled_transform_preserve_nan(
                        transform_input,
                        current_power.lambdas,
                    )
                    - current_power.mean
                )
                / current_power.scale
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_output = (
            compiled_transform_preserve_nan(
                transform_input,
                current_power.lambdas,
            )
            - current_power.mean
        ) / current_power.scale
        error = _max_abs_error(compiled_output, current_transform)
        results.append(
            _result(
                processor="Power",
                operation="transform",
                candidate="vectorized_compiled",
                dtype=transform_input.dtype,
                shape=tuple(transform_input.shape),
                characteristics=label,
                timing=timing,
                error=error,
                correct=torch.allclose(
                    compiled_output,
                    current_transform,
                    rtol=2e-6,
                    atol=2e-6,
                    equal_nan=True,
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    power_inverse_input = numeric.replace_blocks(numerical=current_transform)
    current_power_inverse = current_power.inverse_transform(
        power_inverse_input
    ).numerical
    timing = _measure_cuda(
        lambda: current_power.inverse_transform(power_inverse_input).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Power",
            operation="inverse_transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _vectorized_power_inverse(
            power_inverse_input.numerical,
            current_power.lambdas,
            current_power.mean,
            current_power.scale,
            current_power.upper_bound,
            current_power.max,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    power_inverse = _vectorized_power_inverse(
        power_inverse_input.numerical,
        current_power.lambdas,
        current_power.mean,
        current_power.scale,
        current_power.upper_bound,
        current_power.max,
    )
    error = _max_abs_error(power_inverse, current_power_inverse)
    results.append(
        _result(
            processor="Power",
            operation="inverse_transform",
            candidate="vectorized_eager",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                power_inverse,
                current_power_inverse,
                rtol=1e-3,
                atol=POWER_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    if compile_candidates:
        compiled_power_inverse = torch.compile(
            _vectorized_power_inverse,
            fullgraph=True,
        )
        compiled_power_inverse(
            power_inverse_input.numerical,
            current_power.lambdas,
            current_power.mean,
            current_power.scale,
            current_power.upper_bound,
            current_power.max,
        )
        torch.cuda.synchronize(device)
        timing = _measure_cuda(
            lambda: compiled_power_inverse(
                power_inverse_input.numerical,
                current_power.lambdas,
                current_power.mean,
                current_power.scale,
                current_power.upper_bound,
                current_power.max,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_power_inverse_output = compiled_power_inverse(
            power_inverse_input.numerical,
            current_power.lambdas,
            current_power.mean,
            current_power.scale,
            current_power.upper_bound,
            current_power.max,
        )
        error = _max_abs_error(
            compiled_power_inverse_output,
            current_power_inverse,
        )
        results.append(
            _result(
                processor="Power",
                operation="inverse_transform",
                candidate="vectorized_compiled",
                dtype=transform_input.dtype,
                shape=tuple(transform_input.shape),
                characteristics=label,
                timing=timing,
                error=error,
                correct=torch.allclose(
                    compiled_power_inverse_output,
                    current_power_inverse,
                    rtol=1e-3,
                    atol=POWER_OUTPUT_ATOL,
                    equal_nan=True,
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    quantile = Quantile(output_distribution="normal").fit(fit_table)
    timing = _measure_cuda(
        lambda: (
            Quantile(output_distribution="normal")
            .fit_transform(fit_table)
            .numerical
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    current_quantile_fit_transform = (
        Quantile(output_distribution="normal")
        .fit_transform(fit_table)
        .numerical
    )
    results.append(
        _result(
            processor="Quantile",
            operation="fit_transform",
            candidate="current",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    torch.manual_seed(0)
    quantile_fit_indices = torch.randperm(
        fit_input.shape[0],
        device=device,
    )[:10_000]
    timing = _measure_cuda(
        lambda: _quantile_fit_transform_preindexed(
            fit_input,
            quantile.references,
            quantile_fit_indices,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    quantile_fit_transform = _quantile_fit_transform_preindexed(
        fit_input,
        quantile.references,
        quantile_fit_indices,
    )
    error = _max_abs_error(
        quantile_fit_transform,
        current_quantile_fit_transform,
    )
    results.append(
        _result(
            processor="Quantile",
            operation="fit_transform",
            candidate="direct_nanquantile_searchsorted_preindexed",
            dtype=fit_input.dtype,
            shape=tuple(fit_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                quantile_fit_transform,
                current_quantile_fit_transform,
                rtol=1e-6,
                atol=QUANTILE_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    current_quantile_output = quantile.transform(numeric).numerical
    timing = _measure_cuda(
        lambda: quantile.transform(numeric).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Quantile",
            operation="transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _vectorized_quantile_row_major(
            transform_input,
            quantile.quantiles,
            quantile.references,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    quantile_output = _vectorized_quantile_row_major(
        transform_input,
        quantile.quantiles,
        quantile.references,
    )
    error = _max_abs_error(quantile_output, current_quantile_output)
    results.append(
        _result(
            processor="Quantile",
            operation="transform",
            candidate="batched_searchsorted_row_major",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                quantile_output,
                current_quantile_output,
                rtol=1e-6,
                atol=QUANTILE_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    timing = _measure_cuda(
        lambda: _feature_batched_quantile_row_major(
            transform_input,
            quantile.quantiles,
            quantile.references,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    feature_batched_output = _feature_batched_quantile_row_major(
        transform_input,
        quantile.quantiles,
        quantile.references,
    )
    error = _max_abs_error(feature_batched_output, current_quantile_output)
    results.append(
        _result(
            processor="Quantile",
            operation="transform",
            candidate="feature_batched_searchsorted_32",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                feature_batched_output,
                current_quantile_output,
                rtol=1e-6,
                atol=QUANTILE_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    if compile_candidates:
        compiled_quantile = torch.compile(
            _vectorized_quantile_row_major,
            fullgraph=True,
        )
        compiled_quantile(
            transform_input,
            quantile.quantiles,
            quantile.references,
        )
        torch.cuda.synchronize(device)

        def compiled_quantile_preserve_nan() -> Tensor:
            output = compiled_quantile(
                transform_input,
                quantile.quantiles,
                quantile.references,
            )
            return output.masked_fill(transform_input.isnan(), torch.nan)

        timing = _measure_cuda(
            compiled_quantile_preserve_nan,
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_quantile_output = compiled_quantile_preserve_nan()
        error = _max_abs_error(
            compiled_quantile_output,
            current_quantile_output,
        )
        results.append(
            _result(
                processor="Quantile",
                operation="transform",
                candidate="batched_searchsorted_compiled",
                dtype=transform_input.dtype,
                shape=tuple(transform_input.shape),
                characteristics=label,
                timing=timing,
                error=error,
                correct=torch.allclose(
                    compiled_quantile_output,
                    current_quantile_output,
                    rtol=1e-6,
                    atol=QUANTILE_OUTPUT_ATOL,
                    equal_nan=True,
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    column_major_input = transform_input.T.contiguous()
    column_major_quantiles = quantile.quantiles.T.contiguous()
    timing = _measure_cuda(
        lambda: _vectorized_quantile_col_major(
            column_major_input,
            column_major_quantiles,
            quantile.references,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    column_output = _vectorized_quantile_col_major(
        column_major_input,
        column_major_quantiles,
        quantile.references,
    ).T
    error = _max_abs_error(column_output, current_quantile_output)
    results.append(
        _result(
            processor="Quantile",
            operation="transform",
            candidate="batched_searchsorted_column_major",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                column_output,
                current_quantile_output,
                rtol=1e-6,
                atol=QUANTILE_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )

    for candidate_dtype in (torch.float16, torch.bfloat16, torch.float64):
        typed_input = transform_input.to(dtype=candidate_dtype)
        typed_quantiles = quantile.quantiles.to(dtype=candidate_dtype)
        typed_references = quantile.references.to(dtype=candidate_dtype)
        if candidate_dtype == torch.float64:

            def operation(
                inp: Tensor = typed_input,
                quantiles: Tensor = typed_quantiles,
                references: Tensor = typed_references,
            ) -> Tensor:
                return _vectorized_quantile_row_major(
                    inp,
                    quantiles,
                    references,
                )

        else:

            def operation(
                inp: Tensor = typed_input,
                quantiles: Tensor = typed_quantiles,
                references: Tensor = typed_references,
            ) -> Tensor:
                return _mixed_precision_quantile_row_major(
                    inp,
                    quantiles,
                    references,
                )

        timing = _measure_cuda(
            operation,
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        typed_output = operation().float()
        error = _max_abs_error(typed_output, current_quantile_output)
        results.append(
            _result(
                processor="Quantile",
                operation="transform",
                candidate=f"batched_searchsorted_{candidate_dtype}",
                dtype=candidate_dtype,
                shape=tuple(transform_input.shape),
                characteristics=label,
                timing=timing,
                error=error,
                correct=torch.allclose(
                    typed_output,
                    current_quantile_output,
                    rtol=1e-6,
                    atol=QUANTILE_OUTPUT_ATOL,
                    equal_nan=True,
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    quantile_inverse_input = numeric.replace_blocks(
        numerical=current_quantile_output
    )
    current_quantile_inverse = quantile.inverse_transform(
        quantile_inverse_input
    ).numerical
    timing = _measure_cuda(
        lambda: quantile.inverse_transform(quantile_inverse_input).numerical,
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    results.append(
        _result(
            processor="Quantile",
            operation="inverse_transform",
            candidate="current",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=0.0,
            correct=True,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    timing = _measure_cuda(
        lambda: _batched_inverse_quantile_row_major(
            quantile_inverse_input.numerical,
            quantile.quantiles,
            quantile.references,
            normal=True,
        ),
        device=device,
        repetitions=repetitions,
        warmups=warmups,
    )
    quantile_inverse = _batched_inverse_quantile_row_major(
        quantile_inverse_input.numerical,
        quantile.quantiles,
        quantile.references,
        normal=True,
    )
    error = _max_abs_error(quantile_inverse, current_quantile_inverse)
    results.append(
        _result(
            processor="Quantile",
            operation="inverse_transform",
            candidate="batched_searchsorted_row_major",
            dtype=transform_input.dtype,
            shape=tuple(transform_input.shape),
            characteristics=label,
            timing=timing,
            error=error,
            correct=torch.allclose(
                quantile_inverse,
                current_quantile_inverse,
                rtol=1e-6,
                atol=QUANTILE_OUTPUT_ATOL,
                equal_nan=True,
            ),
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    )
    if compile_candidates:
        compiled_quantile_inverse = torch.compile(
            _batched_inverse_quantile_row_major,
            fullgraph=True,
        )
        compiled_quantile_inverse(
            quantile_inverse_input.numerical,
            quantile.quantiles,
            quantile.references,
            normal=True,
        )
        torch.cuda.synchronize(device)
        timing = _measure_cuda(
            lambda: compiled_quantile_inverse(
                quantile_inverse_input.numerical,
                quantile.quantiles,
                quantile.references,
                normal=True,
            ),
            device=device,
            repetitions=repetitions,
            warmups=warmups,
        )
        compiled_quantile_inverse_output = compiled_quantile_inverse(
            quantile_inverse_input.numerical,
            quantile.quantiles,
            quantile.references,
            normal=True,
        )
        error = _max_abs_error(
            compiled_quantile_inverse_output,
            current_quantile_inverse,
        )
        results.append(
            _result(
                processor="Quantile",
                operation="inverse_transform",
                candidate="batched_searchsorted_compiled",
                dtype=transform_input.dtype,
                shape=tuple(transform_input.shape),
                characteristics=label,
                timing=timing,
                error=error,
                correct=torch.allclose(
                    compiled_quantile_inverse_output,
                    current_quantile_inverse,
                    rtol=1e-6,
                    atol=QUANTILE_OUTPUT_ATOL,
                    equal_nan=True,
                ),
                repetitions=repetitions,
                warmups=warmups,
                device=device,
            )
        )

    transfer_specs = (
        (
            "Power",
            "fit",
            tuple(fit_input.shape),
            (5, fit_input.shape[1]),
        ),
        (
            "Quantile",
            "fit",
            tuple(fit_input.shape),
            (quantile.references.numel(), fit_input.shape[1]),
        ),
    )
    for processor in ("Power", "Quantile"):
        transfer_specs += (
            (
                processor,
                "fit_transform",
                tuple(fit_input.shape),
                tuple(fit_input.shape),
            ),
            (
                processor,
                "transform",
                tuple(transform_input.shape),
                tuple(transform_input.shape),
            ),
            (
                processor,
                "inverse_transform",
                tuple(transform_input.shape),
                tuple(transform_input.shape),
            ),
        )
    for processor, operation, input_shape, output_shape in transfer_specs:
        _append_transfer_results(
            results,
            processor=processor,
            operation=operation,
            input_shape=input_shape,
            output_shape=output_shape,
            characteristics=label,
            repetitions=repetitions,
            warmups=warmups,
            device=device,
        )
    return results


def _write_results(results: Sequence[Result], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    sync_median, sync_p95 = _idle_synchronize_timing(device)
    payload = {
        "environment": {
            "gpu_model": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "idle_synchronize_median_ms": sync_median,
            "idle_synchronize_p95_ms": sync_p95,
        },
        "results": [asdict(result) for result in results],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=asdict(results[0]).keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    """Run and persist the GPU speed-of-light benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/results/processor_gpu_speed_of_light.json"),
    )
    args = parser.parse_args()
    results = run(
        repetitions=args.repetitions,
        warmups=args.warmups,
        compile_candidates=not args.no_compile,
    )
    _write_results(results, args.output)
    for result in results:
        print(  # noqa: T201
            f"{result.processor}.{result.operation} {result.candidate}: "
            f"{result.median_ms:.3f} ms "
            f"(CUDA {result.cuda_median_ms:.3f} ms, "
            f"{result.correctness_status})"
        )


if __name__ == "__main__":
    main()
