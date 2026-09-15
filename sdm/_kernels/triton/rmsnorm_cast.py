# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, cast

import torch
from torch import Tensor

if sys.platform != "linux":
    raise ImportError("Triton kernels are only available on Linux")

import triton
import triton.language as tl


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
    # Preserve native RMSNorm's accumulation order before the final cast.
    for i in tl.static_range(4):
        v = tl.gather(value, groups * 4 + i, axis=0)
        total = tl.fma(v, v, total)
    variance = tl.sum(total, 0)  # ty: ignore[invalid-argument-type]
    value *= tl.rsqrt(variance / channels + eps)
    value *= tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + row * channels + c, value)


def rmsnorm_cast(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x = x.contiguous()
    channels = x.size(-1)
    out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda"))
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
