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
            the number of columns.
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
            columns, ``N`` is the number of patches, and ``P`` is the patch
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
    x: Tensor,  # [..., N, P]
    num_patches: int,
) -> tuple[Tensor, Tensor]:  # [..., N, num_patches * P]
    """Concatenate the next future patches at each patch position.

    Args:
        x: Input with shape ``[..., N, P]``, where ``N`` is the number of
            patches, and ``P`` is the patch size.
        num_patches: Number of following patches to Concatenate.

    Returns:
        Rolled values with shape ``[..., N, num_patches * P]`` and a
        broadcastable mask marking invalid wrapped values.
    """
    *B, N, P = x.size()

    offset = torch.arange(1, num_patches + 1, device=x.device)
    index = torch.arange(N, device=x.device).unsqueeze(-1) + offset

    out = x[..., index.view(-1) % N, :]  # [..., N * num_patches, P]
    out = out.view(*B, N, num_patches * P)

    mask = (index >= N).repeat_interleave(P, dim=-1)
    mask = mask.view(*(1,) * (x.dim() - 2), *mask.size())

    return out, mask


def crossfade_patches(
    x: Tensor,  # [..., N, P, ...]
    step: int,
    dim: int,
) -> Tensor:  # [..., (N - 1) * step + P, ...]
    """Merge overlapping patches into a single sequence, crossfading overlaps.

    Patch ``i`` starts at position ``i * step``. Whenever consecutive patches
    overlap, values are linearly interpolated from the earlier patch to the
    later one.

    Args:
        x: Input with shape ``[..., N, P, ...]``, where ``N`` is the
            number of patches, and ``P`` is the patch size.
        step: Distance between consecutive patch starts across ``N``.
        dim: The patch dimension ``N``.

    Returns:
        Merged sequence with shape ``[..., (N - 1) * step + P, ...]``.
    """
    dim = dim % x.dim()
    N, P = x.size(dim), x.size(dim + 1)

    overlap = P - step
    assert 0 <= overlap <= step

    weight_shape = [1] * x.dim()
    weight_shape[dim + 1] = overlap
    weight = torch.linspace(1.0, 0.0, overlap, device=x.device, dtype=x.dtype)
    weight = weight.view(weight_shape)

    tail = x.narrow(dim, 0, N - 1).narrow(dim + 1, step, overlap)
    head = x.narrow(dim, 1, N - 1).narrow(dim + 1, 0, overlap)
    blended = weight * tail + (1.0 - weight) * head
    body = x.narrow(dim, 1, N - 1).narrow(dim + 1, overlap, step - overlap)

    first = x.select(dim, 0).narrow(dim, 0, step)
    middle = torch.cat([blended, body], dim=dim + 1)
    middle = middle.flatten(dim, dim + 1)
    last = x.select(dim, N - 1).narrow(dim, step, overlap)

    return torch.cat([first, middle, last], dim=dim)
