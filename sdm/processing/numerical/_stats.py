# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor


def _isfinite(x: Tensor) -> Tensor:
    # Equal to 'x.isfinite()', which allocates 'x.abs()' on the way.
    return x.gt(-math.inf).logical_and_(x.lt(math.inf))


def _count(mask: Tensor) -> Tensor:
    # [..., N, C] -> [..., 1, C] int64 number of true values per column.
    # Summing bool first casts all of 'mask' to int64. Sum blocks of 255 rows
    # as uint8 instead, which cannot overflow.
    num_blocks = mask.size(-2) // 255
    blocks = mask[..., : num_blocks * 255, :].unflatten(-2, (num_blocks, 255))
    count = blocks.view(torch.uint8).sum(-2, dtype=torch.uint8)
    remainder = mask[..., num_blocks * 255 :, :]
    return count.sum(-2, keepdim=True) + remainder.sum(-2, keepdim=True)


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    num_samples: int | Tensor,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = num_samples * eps * var + (num_samples * mean * eps) ** 2
    return var <= upper_bound
