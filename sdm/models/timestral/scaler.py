# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
from torch import Tensor


def _fp32_io(*dtypes: torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    out_dtype = functools.reduce(torch.promote_types, dtypes)
    return (out_dtype, torch.promote_types(out_dtype, torch.float32))


def _apply_affine_asinh_masked(
    data: Tensor,
    loc: Tensor,
    scale: Tensor,
    observation_mask: Tensor,
    use_asinh: bool,
) -> Tensor:
    out_dtype, work = _fp32_io(data.dtype, loc.dtype, scale.dtype)
    scaled = (data.to(work) - loc.to(work)) / scale.to(work).clamp_min(
        torch.finfo(work).eps
    )
    scaled = scaled.masked_fill(~observation_mask, 0.0)
    if use_asinh:
        scaled = torch.where(observation_mask, scaled.asinh(), scaled)
    return scaled.to(out_dtype)


def _invert_affine_asinh(
    x: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool
) -> Tensor:
    out_dtype, work = _fp32_io(x.dtype, loc.dtype, scale.dtype)
    x = x.to(work)
    if use_asinh:
        x = x.sinh()
    return (x * scale.to(work) + loc.to(work)).to(out_dtype)


def _clamp_nonfinite(vals: Tensor) -> Tensor:
    clamped = torch.where(
        vals == float("inf"),
        torch.where(vals.isfinite(), vals, -float("inf")).amax(
            dim=-1, keepdim=True
        ),
        torch.where(
            vals == -float("inf"),
            torch.where(vals.isfinite(), vals, float("inf")).amin(
                dim=-1, keepdim=True
            ),
            vals,
        ),
    )
    return torch.where(clamped.isinf(), torch.zeros_like(clamped), clamped)


def _denormalize_quantiles(
    preds: Tensor, loc: Tensor, scale: Tensor, use_asinh: bool
) -> Tensor:
    return _clamp_nonfinite(_invert_affine_asinh(preds, loc, scale, use_asinh))


def _welford_combine(
    w_a: Tensor,
    mean_a: Tensor,
    m2_a: Tensor,
    w_b: Tensor,
    mean_b: Tensor,
    m2_b: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    w = w_a + w_b
    w_safe = w.clamp(min=torch.finfo(w.dtype).tiny)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + delta.square() * w_a * w_b / w_safe
    is_zero = w == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    return (w, mean, m2)


def _cumulative_welford(
    w: Tensor, mean: Tensor, m2: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    n = w.shape[1]
    d = 1
    while d < n:
        pad = (d, 0)
        w_l = torch.nn.functional.pad(w[:, : n - d], pad)
        mean_l = torch.nn.functional.pad(mean[:, : n - d], pad)
        m2_l = torch.nn.functional.pad(m2[:, : n - d], pad)
        w, mean, m2 = _welford_combine(w_l, mean_l, m2_l, w, mean, m2)
        d *= 2
    return (w, mean, m2)


def _advancing_block_frames(
    context: Tensor,
    observation_mask: Tensor,
    sinh_median: Tensor,
    minimum_scale: float,
) -> tuple[Tensor, Tensor]:
    r, n_pred, ps = sinh_median.shape
    blank = sinh_median.new_zeros(r, n_pred, ps)
    values = torch.cat(
        [context.unflatten(-1, (-1, ps)).to(torch.float64), blank], dim=1
    )
    weights = torch.cat(
        [observation_mask.unflatten(-1, (-1, ps)).to(torch.float64), blank],
        dim=1,
    )
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / cnt.clamp(min=1.0)
    m2 = (weights * (values - mean.unsqueeze(-1)).square()).sum(-1)
    is_zero = cnt == 0
    mean = torch.where(is_zero, torch.zeros_like(mean), mean)
    m2 = torch.where(is_zero, torch.zeros_like(m2), m2)
    cnt, mean, m2 = (
        s[:, -n_pred - 1] for s in _cumulative_welford(cnt, mean, m2)
    )
    offset = mean
    clip = torch.finfo(torch.float32).max ** 0.5 / 1000.0
    locs, scales = ([], [])
    for h in range(n_pred):
        var = (m2 / (cnt - 1.0).clamp(min=1.0)).clamp(min=0.0)
        scale = var.sqrt().clamp(min=minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (
            sinh_median[:, h] * scale[:, None]
            + mean[:, None]
            - offset[:, None]
        ).clamp(-clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = (folded - b_mean[:, None]).square().sum(-1)
        cnt, mean, m2 = _welford_combine(
            w_a=cnt,
            mean_a=mean,
            m2_a=m2,
            w_b=torch.full_like(cnt, ps),
            mean_b=b_mean,
            m2_b=b_m2,
        )
    return (torch.stack(locs, dim=1), torch.stack(scales, dim=1))


def _instance_norm_stats(
    values: Tensor, observation_mask: Tensor, minimum_scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    orig_dtype = values.dtype
    stat_dtype = torch.promote_types(orig_dtype, torch.float32)
    x = values.to(stat_dtype)
    mask_f = observation_mask.to(stat_dtype)
    n = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
    loc = (x * mask_f).sum(dim=-1, keepdim=True) / n
    var = (((x - loc) * mask_f) ** 2).sum(dim=-1, keepdim=True) / n
    scale = var.sqrt().clamp_min(minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(
        dim=-1, keepdim=True
    )
    is_binary = is_zero_one & observation_mask.any(dim=-1, keepdim=True)
    loc = torch.where(is_binary, torch.zeros_like(loc), loc)
    scale = torch.where(is_binary, torch.ones_like(scale), scale)
    return (loc.to(orig_dtype), scale.to(orig_dtype), is_binary)
