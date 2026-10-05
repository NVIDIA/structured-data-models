# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def update_running_stats(
    count: Tensor,  # [..., C]
    mean: Tensor,  # [..., C]
    std: Tensor,  # [..., C]
    x: Tensor,  # [..., C, P]
    mask: Tensor,  # [..., C, P]
) -> tuple[Tensor, Tensor, Tensor]:  # [..., C]
    """Update running statistics for a new patch.

    Args:
        count: Count of valid values with shape ``[..., C]``, where ``C`` is
            the number of channels.
        mean: Running mean with shape ``[..., C]``.
        std: Running standard deviation with shape ``[..., C]``.
        x: New input with shape ``[..., C, P]``, where ``P`` is the patch size.
        mask: Invalid-value mask with shape ``[..., C, P]``.

    Returns:
        Updated count, mean, and standard deviation.
    """
    valid = ~mask

    inc_count = valid.sum(dim=-1)
    inc_sum = torch.where(valid, x, 0.0).sum(dim=-1)
    inc_mean = inc_sum / inc_count.clamp(min=1)
    tmp = (x - inc_mean.unsqueeze(-1)).square()
    inc_var = torch.where(valid, tmp, 0.0).sum(dim=-1) / inc_count.clamp(min=1)
    inc_std = inc_var.sqrt()

    out_count = count + inc_count
    out_mean = (count * mean + inc_count * inc_mean) / out_count.clamp(min=1)
    out_var = (
        count * std.square()
        + inc_count * inc_std.square()
        + count * (mean - out_mean).square()
        + inc_count * (inc_mean - out_mean).square()
    ) / out_count.clamp(min=1)
    out_std = out_var.sqrt()

    return out_count, out_mean, out_std


def get_running_stats(
    x: Tensor,  # [..., C, N, P]
    mask: Tensor,  # [..., C, N, P]
) -> tuple[Tensor, Tensor, Tensor]:  # [..., C, N]
    """Compute cumulative statistics patch by patch.

    Args:
        x: Input with shape ``[..., C, N, P]``, where ``C`` is the number of
            channels, ``N`` is the number of patches, and ``P`` is the patch
            size.
        mask: Invalid-value mask with shape ``[..., C, N, P]``.

    Returns:
        Cumulative count, mean, and standard deviation.
    """
    *B, C, N, _ = x.size()

    count = torch.zeros((*B, C), dtype=torch.int64, device=x.device)
    mean = torch.zeros((*B, C), dtype=torch.float32, device=x.device)
    std = torch.zeros((*B, C), dtype=torch.float32, device=x.device)

    counts, means, stds = [], [], []
    for i in range(N):
        count, mean, std = update_running_stats(
            count, mean, std, x[..., i, :], mask[..., i, :]
        )
        counts.append(count)
        means.append(mean)
        stds.append(std)

    return (
        torch.stack(counts, dim=-1),
        torch.stack(means, dim=-1),
        torch.stack(stds, dim=-1),
    )
  
def revin(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    reverse: bool = False,
) -> Tensor:
    r"""Apply the statistical transform described by the `RevIN paper`_.

    This helper uses supplied statistics and omits RevIN's learnable affine
    transformation.

    Args:
        x: Input values with shape ``[..., D]`` or ``[..., D, Q]``, where
            ``D`` is the number of values and ``Q`` is the optional number of
            prediction channels.
        mu: Means with one or two fewer dimensions than ``x``.
        sigma: Standard deviations with the same shape as ``mu``.
        reverse: Whether to denormalize instead of normalize.

    Returns:
        Transformed values with the same shape as ``x``.

    .. _RevIN paper: https://openreview.net/forum?id=cGDAkQo1C0p
    """
    if mu.shape != sigma.shape:
        raise ValueError(
            "mu and sigma must have the same shape, got "
            f"{mu.shape} and {sigma.shape}."
        )

    if mu.dim() not in (x.dim() - 1, x.dim() - 2):
        raise ValueError(
            f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}."
        )
    if mu.shape != x.shape[: mu.dim()]:
        raise ValueError(
            "mu and sigma must match the leading dimensions of x, got "
            f"x.shape={x.shape} and stats shape={mu.shape}."
        )

    if mu.dim() == x.dim() - 1:
        mu = mu.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    else:
        mu = mu.unsqueeze(-1).unsqueeze(-1)
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)

    if reverse:
        return x * sigma + mu
    return (x - mu) / _make_safe_for_division(sigma)
