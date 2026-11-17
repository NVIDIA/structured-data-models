# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor


def cdist_argmin(x: Tensor, y: Tensor, p: float) -> Tensor:
    """Return nearest-center indices using Minkowski distance.

    General exponents are evaluated in row chunks. Near ties use
    :func:`torch.cdist` to resolve the nearest center.

    Args:
        x: Input rows with shape ``[..., R, H]`` for rows and channels.
        y: Center values with shape ``[..., K, H]`` for centers and channels.
        p: Minkowski distance exponent.

    Returns:
        Center indices with shape ``[..., R]``.
    """
    # Only `torch.cdist` broadcasts batch dimensions of different sizes.
    if (
        p in (1.0, 2.0)
        or not 0.0 < p < math.inf
        or x.shape[:-2] != y.shape[:-2]
    ):
        return torch.cdist(x, y, p=p).argmin(dim=-1)
    if x.dim() > 2:
        return torch.stack([cdist_argmin(a, b, p) for a, b in zip(x, y)])

    # Equal rows of `y` give equal sums, which the tie test would send to
    # `torch.cdist`. Compare one row of each group, and return the first
    # index of the group, as `torch.cdist` does.
    unique, group = torch.unique(y, dim=0, return_inverse=True)
    first = group.new_full((unique.size(0),), y.size(0))
    order = torch.arange(y.size(0), device=y.device)
    first.scatter_reduce_(0, group, order, "amin")

    num_rows, dims = x.shape
    # The tie test allows for the rounding of both sums. The power differs
    # from `torch.cdist` by a relative 1e-5 at most, and a sum of `dims`
    # terms by `dims * eps`. log2 is slow on zero, so the power raises
    # values below `tiny` to `tiny`. Below `tiny`, floats have a fixed step
    # of `tiny * eps`, so rounding can differ by one step for each column.
    info = torch.finfo(x.dtype)
    tiny, eps = info.tiny, info.eps
    scale = 1.0 + 1e-4 + 2.0 * dims * eps
    slack = 2.0 * dims * (tiny**p + tiny * eps)
    index = torch.empty(num_rows, dtype=torch.long, device=x.device)
    tie = torch.empty(num_rows, dtype=torch.bool, device=x.device)
    chunk = max(1, 2**20 // (unique.size(0) * dims))  # Stays in the cache.
    # A new tensor for each chunk is slower on a node with many workers.
    buffer = x.new_empty(min(chunk, num_rows), unique.size(0), dims)
    for start in range(0, num_rows, chunk):
        stop = min(start + chunk, num_rows)
        diff = buffer[: stop - start]
        torch.sub(x[start:stop].unsqueeze(1), unique, out=diff)
        dist = diff.abs_().clamp_min_(tiny).log2_().mul_(p).exp2_().sum(-1)
        best, nearest = dist.min(dim=-1)
        index[start : start + chunk] = nearest
        bound = best.unsqueeze(-1) * scale + slack
        near = (dist <= bound).sum(dim=-1) > 1
        tie[start : start + chunk] = near | best.isnan()

    out = first[index]
    rows = tie.nonzero().squeeze(-1)
    if rows.numel() > 0:
        out[rows] = torch.cdist(x[rows], y, p=p).argmin(dim=-1)
    return out
