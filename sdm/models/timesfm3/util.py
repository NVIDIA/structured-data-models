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
    mean: Tensor,
    std: Tensor,
    reverse: bool = False,
) -> Tensor:
    r"""Apply the reversible instance normalization from the
    `"Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift" <https://openreview.net/forum?id=cGDAkQo1C0p>`_
    paper.

    Args:
        x: Input values with shape ``[..., C]`` or ``[..., C_1, C_2]``.
        mean: Means with shape ``[...]``.
        std: Standard deviation with shape ``[...]``.
        reverse: Whether to denormalize rather than normalize.
    """  # noqa: D205
    assert mean.dim() == std.dim()
    assert mean.dim() + 2 >= x.dim()
    for _ in range(x.dim() - mean.dim()):
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)

    if reverse:
        return x * std + mean

    return (x - mean) / torch.where(std < 1e-6, 1.0, std)


def gather_future_patches(
    x: Tensor,  # [..., C, N, P]
    num_future_patches: int,
) -> tuple[Tensor, Tensor]:  # [..., C, N, K * P], [1, 1, N, K * P]
    """Concatenate next patches to each patch, wrapping around past the end.

    Args:
        x: Input patches with shape ``[..., C, N, P]``, where ``C`` is the
            number of channels, ``N`` is the number of patches, and ``P`` is
            the patch size.
        num_future_patches: Number ``K`` of subsequent patches to concatenate.

    Returns:
        Concatenated patches with shape ``[..., C, N, K * P]``, and a
        wrap-around mask with shape ``[1, 1, N, K * P]`` marking values taken
        from the beginning of the sequence.
    """
    *B, C, N, P = x.size()
    K = num_future_patches

    offsets = torch.arange(1, K + 1, device=x.device)
    indices = torch.arange(N, device=x.device).unsqueeze(-1) + offsets

    future = x.index_select(-2, (indices % N).flatten())
    future = future.reshape(*B, C, N, K * P)
    mask = (indices >= N).repeat_interleave(P, dim=-1)

    return future, mask.unsqueeze(0).unsqueeze(0)


def crossfade_patches(
    patches: Tensor,  # [..., N, S + O, D]
    step: int,
) -> Tensor:  # [..., N * S + O, D]
    """Merge overlapping patches into a sequence, crossfading the overlaps.

    Patch ``i`` starts at position ``i * S``. Where consecutive patches
    overlap, values are linearly interpolated from the earlier patch to the
    later one.

    Args:
        patches: Patches with shape ``[..., N, S + O, D]``, where ``N`` is the
            number of patches, ``O`` is the overlap between consecutive
            patches, and ``D`` is the feature dimension.
        step: Distance ``S`` between patch starts, at least ``O``.

    Returns:
        Merged sequence with shape ``[..., N * S + O, D]``.
    """
    *B, N, T, D = patches.size()
    if N == 1:
        return patches[..., 0, :, :]

    overlap = T - step
    assert 0 <= overlap <= step

    weights = torch.linspace(
        1.0, 0.0, overlap, device=patches.device, dtype=patches.dtype
    ).unsqueeze(-1)

    tails = patches[..., :-1, step:, :]
    heads = patches[..., 1:, :overlap, :]
    blended = weights * tails + (1.0 - weights) * heads
    bodies = patches[..., 1:, overlap:step, :]

    first = patches[..., 0, :step, :]
    middle = torch.cat((blended, bodies), dim=-2)
    middle = middle.reshape(*B, (N - 1) * step, D)
    last = patches[..., -1, step:, :]

    return torch.cat((first, middle, last), dim=-2)
