# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    channels: tl.constexpr,
    eps: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, channels)
    value = tl.load(x_ptr + row * channels + c).to(tl.float32)
    groups = tl.arange(0, channels // 4)
    total = tl.full((channels // 4,), 0, tl.float32)
    for i in tl.static_range(4):
        v = tl.gather(value, groups * 4 + i, axis=0)
        total = tl.fma(v, v, total)
    variance = tl.sum(total, 0)  # ty: ignore[invalid-argument-type]
    value *= tl.rsqrt(variance / channels + eps)
    if weight_ptr is not None:
        value *= tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + row * channels + c, value)


def rmsnorm_cast(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    x = x.contiguous()
    if weight is not None:
        weight = weight.contiguous()
    channels = x.size(-1)
    out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda"))
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_cast_kernel)[(x.numel() // channels,)](
            x,
            weight,
            out,
            channels,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
