# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

from sdm._kernels.triton.rmsnorm_cast import (
    _load_input,
    _row_size_arguments,
    _variance,
)


@triton.jit(
    do_not_specialize=[
        "rows",
        "row_sizes",
        "row_strides",
        "size0",
        "size1",
        "size2",
        "channel_stride",
    ]
)
def _rmsnorm_pair_kernel(
    X,
    WA,
    WB,
    A,
    B,
    rows,
    row_sizes,
    row_strides,
    size0,
    size1,
    size2,
    channel_stride,
    C: tl.constexpr,
    EA: tl.constexpr,
    EB: tl.constexpr,
    R: tl.constexpr,
) -> None:
    if len(row_sizes) == 0:
        row_sizes = (size0, size1, size2)[: len(row_strides)]
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    offsets = row[:, None] * C
    if len(row_sizes) == 0:
        input_offsets = row * C
        channel_stride = 1
    else:
        remaining = row
        input_offsets = tl.full((R,), 0, tl.int64)
        for axis in tl.static_range(len(row_sizes) - 1, -1, -1):
            if axis == 0:
                input_offsets += remaining * row_strides[axis]
            else:
                input_offsets += (
                    remaining % row_sizes[axis] * row_strides[axis]
                )
                remaining //= row_sizes[axis]
    mask = row[:, None] < rows
    variance = _variance(  # ty: ignore[invalid-argument-type]
        X,
        None,
        None,
        row,
        input_offsets,
        channel_stride,
        mask,
        C,
        1,
        1,
        R,
        len(row_sizes) > 0,  # ty: ignore[invalid-argument-type]
    )
    c = tl.arange(0, C)[None, :]
    value = _load_input(  # ty: ignore[invalid-argument-type]
        X,
        None,
        None,
        row,
        input_offsets,
        channel_stride,
        c,
        mask,
        C,
        1,
        1,
    )
    a = value * tl.rsqrt(variance[:, None] / C + EA)
    b = value * tl.rsqrt(variance[:, None] / C + EB)
    if WA is not None:
        a *= tl.load(WA + c).to(tl.float32)
    if WB is not None:
        b *= tl.load(WB + c).to(tl.float32)
    tl.store(A + offsets + c, a, mask)
    tl.store(B + offsets + c, b, mask)


def rmsnorm_pair(
    x: Tensor,
    wa: Tensor | None,
    wb: Tensor | None,
    ea: float,
    eb: float,
) -> tuple[Tensor, Tensor]:
    dtype = torch.get_autocast_dtype("cuda")
    row_sizes = () if x.is_contiguous() else (x.shape[:-1] or (1,))
    row_strides = () if not row_sizes else (x.stride()[:-1] or (0,))
    row_sizes, sizes = _row_size_arguments(row_sizes)
    wa = wa.contiguous() if wa is not None else None
    wb = wb.contiguous() if wb is not None else None
    rows, channels = x.numel() // x.size(-1), x.size(-1)
    a = torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)
    b = torch.empty_like(a)
    with torch.cuda.device(x.device):
        cast(Any, _rmsnorm_pair_kernel)[((rows + 7) // 8,)](
            x,
            wa,
            wb,
            a,
            b,
            rows,
            row_sizes,
            row_strides,
            *sizes,
            x.stride(-1),
            channels,
            ea,
            eb,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return a, b
