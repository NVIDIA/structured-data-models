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
