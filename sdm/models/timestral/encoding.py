# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor


def _encode_categories(
    target: Tensor, codes: Tensor, max_context: int, num_categories: int
) -> Tensor:
    b, c, _length = codes.shape
    t = target.size(-1)
    valid = target.isfinite()
    valid &= torch.arange(t, device=target.device) >= t - max_context
    y = target.double().nan_to_num()
    mean = (y * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    # Each column has an extra slot for missing values.
    slots = num_categories + 1
    rows = torch.arange(b * c, device=codes.device).view(b, c, 1)
    index = (rows * slots + codes.clamp_min(-1) + 1).expand_as(codes)
    sums = y.new_zeros(b * c * slots)
    counts = y.new_zeros(b * c * slots)
    sums.scatter_add_(
        0,
        index[..., :t].reshape(-1),
        (y[:, None] * valid[:, None]).expand(b, c, t).reshape(-1),
    )
    counts.scatter_add_(
        0,
        index[..., :t].reshape(-1),
        valid[:, None].expand(b, c, t).double().reshape(-1),
    )
    prior = mean[:, None, None].expand(b, c, slots).reshape(-1)
    out = ((prior + sums) / (1 + counts))[index].float()
    out[..., :t] = out[..., :t].masked_fill(
        ~valid[:, None] & (codes[..., :t] < 0), torch.nan
    )
    return out
