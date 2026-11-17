# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def _nanstd(
    x: Tensor,
    dim: int,
    keepdim: bool,
    ddof: int = 0,
) -> Tensor:
    valid = ~torch.isnan(x)
    n = valid.sum(dim=dim, keepdim=True)
    mean = torch.nansum(x, dim=dim, keepdim=True) / n.clamp(min=1)
    var = torch.nansum((x - mean) ** 2, dim=dim, keepdim=True) / (
        n - ddof
    ).clamp(min=1)
    std = var.sqrt()
    return std if keepdim else std.squeeze(dim)


def _clip_outliers(x: Tensor, threshold: float = 4.0) -> Tensor:
    dim = -2
    ddof = 1 if x.shape[dim] > 1 else 0
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    std = _nanstd(x, dim=dim, keepdim=True, ddof=ddof).clamp(min=1e-6)
    lower, upper = mean - threshold * std, mean + threshold * std

    inliers = torch.where(
        condition=(lower <= x) & (x <= upper) & ~torch.isnan(x),
        input=x,
        other=torch.nan,
    )
    inlier_mean = torch.nanmean(inliers, dim=dim, keepdim=True)
    inlier_std = _nanstd(inliers, dim=dim, keepdim=True, ddof=ddof).clamp(
        min=1e-6
    )
    inlier_mean = torch.where(torch.isnan(inlier_mean), mean, inlier_mean)
    inlier_std = torch.where(
        condition=torch.isnan(inlier_std),
        input=torch.zeros_like(std),
        other=inlier_std,
    )

    cutoff = threshold * inlier_std
    lower = torch.nan_to_num(inlier_mean - cutoff, nan=-torch.inf)
    upper = torch.nan_to_num(inlier_mean + cutoff, nan=torch.inf)
    return x.clamp(min=lower, max=upper)


def _standardize(x: Tensor, clip: float = 100.0) -> Tensor:
    dim = -2
    ddof = 1 if x.shape[dim] > 1 else 0
    mean = torch.nanmean(x, dim=dim, keepdim=True)
    std = _nanstd(x, dim=dim, keepdim=True, ddof=ddof).clamp(min=1e-6)
    return ((x - mean) / std).clamp(-clip, clip)
