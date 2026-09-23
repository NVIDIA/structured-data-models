# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit(do_not_specialize=["rows"])
def _rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    offsets = row[:, None] * channels
    mask = row[:, None] < rows
    groups = tl.arange(0, channels // 4)[None, :]
    total = tl.full((block_rows, channels // 4), 0, tl.float32)
    for i in tl.static_range(4):
        v = tl.load(x_ptr + offsets + groups * 4 + i, mask=mask, other=0).to(
            tl.float32
        )
        total = tl.fma(v, v, total)
    variance = tl.sum(total, 1)  # ty: ignore[invalid-argument-type]
    c = tl.arange(0, channels)[None, :]
    value = tl.load(x_ptr + offsets + c, mask=mask, other=0).to(tl.float32)
    value *= tl.rsqrt(variance[:, None] / channels + eps)
    value *= tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + offsets + c, value, mask=mask)


def rmsnorm_cast(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    channels = x.size(-1)
    rows = x.numel() // channels
    block_rows = 8
    grid = ((rows + block_rows - 1) // block_rows,)
    out = torch.empty_like(x, dtype=torch.get_autocast_dtype("cuda"))
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_cast_kernel)[grid](
            x,
            weight,
            out,
            rows,
            channels,
            eps,
            block_rows,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
