# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
from torch import Tensor

from sdm.nn import RevIN


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


def _advancing_block_frames(
    context: Tensor,
    observation_mask: Tensor,
    sinh_median: Tensor,
    minimum_scale: float,
) -> tuple[Tensor, Tensor]:
    _, n_pred, ps = sinh_median.shape
    state = RevIN.running_stats(
        x=context.unsqueeze(-2),
        mask=~observation_mask.unsqueeze(-2),
        correction=1,
    )
    offset = state[0][:, 0, :]
    clip = torch.finfo(torch.float32).max ** 0.5 / 1000.0
    locs, scales = ([], [])
    for h in range(n_pred):
        mean = state[0][:, 0, 0]
        scale = state[1][:, 0, 0].clamp_min(minimum_scale)
        locs.append(mean)
        scales.append(scale)
        folded = (
            sinh_median[:, h] * scale[:, None] + mean[:, None] - offset
        ).clamp(-clip, clip)
        folded = folded + offset
        state = RevIN.running_stats(
            x=folded.unsqueeze(-2),
            state=state,
            correction=1,
        )
    return (torch.stack(locs, dim=1), torch.stack(scales, dim=1))


def _instance_norm_stats(
    values: Tensor, observation_mask: Tensor, minimum_scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    orig_dtype = values.dtype
    loc, scale = RevIN.stats(values, mask=~observation_mask, correction=0)
    observed = observation_mask.any(dim=-1, keepdim=True)
    scale = scale.where(observed, 0.0).clamp_min(minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(
        dim=-1, keepdim=True
    )
    is_binary = is_zero_one & observed
    loc = torch.where(is_binary, torch.zeros_like(loc), loc)
    scale = torch.where(is_binary, torch.ones_like(scale), scale)
    return (loc.to(orig_dtype), scale.to(orig_dtype), is_binary)
