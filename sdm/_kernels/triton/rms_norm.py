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

# Elements normalized by one program.
_BLOCK_SIZE = 4096


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "size1",
        "size2",
        "stride0",
        "stride1",
        "stride2",
        "seq_len",
        "num_heads",
        "eps",
    ]
)
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_rows,
    size1,
    size2,
    stride0,
    stride1,
    stride2,
    seq_len,
    num_heads,
    eps,
    channels: tl.constexpr,
    block_rows: tl.constexpr,
    has_weight: tl.constexpr,
    has_rope: tl.constexpr,
) -> None:
    rows = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(
        0, block_rows
    )
    mask = (rows < num_rows)[:, None]
    c = tl.arange(0, channels)

    offsets = (
        rows // (size1 * size2) * stride0
        + rows // size2 % size1 * stride1
        + rows % size2 * stride2
    )
    x = tl.load(x_ptr + offsets[:, None] + c[None, :], mask=mask, other=0.0)
    x = x.to(tl.float32)
    if has_rope:
        # Split-half rotary embedding, rounded like its eager evaluation in
        # the input dtype.
        dtype = x_ptr.dtype.element_ty
        first = c < channels // 2
        partner = tl.where(first, c + channels // 2, c - channels // 2)
        y = tl.load(
            x_ptr + offsets[:, None] + partner[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        table = (rows // num_heads % seq_len)[:, None] * (channels // 2) + (
            c % (channels // 2)
        )[None, :]
        cos = tl.load(cos_ptr + table, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + table, mask=mask, other=0.0).to(tl.float32)
        a = (x * cos).to(dtype).to(tl.float32)
        b = (y * sin).to(dtype).to(tl.float32)
        x = tl.where(first[None, :], a - b, a + b).to(dtype).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / channels + eps)
    x = x * rstd[:, None]
    if has_weight:
        x = x * tl.load(weight_ptr + c).to(tl.float32)[None, :]
    tl.store(
        out_ptr + rows[:, None] * channels + c[None, :],
        x.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def rms_norm(
    x: Tensor,  # [..., S, H, C] with `rope`, else [..., C]
    weight: Tensor | None,  # [C]
    eps: float,
    dtype: torch.dtype,
    rope: tuple[Tensor, Tensor] | None = None,  # 2 x [S, C // 2]
) -> Tensor | None:  # [..., C]
    channels = x.size(-1)
    if (
        channels & (channels - 1) != 0
        or not 16 <= channels <= _BLOCK_SIZE
        or x.stride(-1) != 1
        or (rope is not None and x.dim() < 3)
    ):
        return None

    # Merge leading dimensions that are contiguous in memory:
    sizes: list[int] = []
    strides: list[int] = []
    for size, stride in zip(x.size()[:-1], x.stride()[:-1], strict=True):
        if size == 1:
            continue
        if sizes and strides[-1] == stride * size:
            sizes[-1] *= size
            strides[-1] = stride
        else:
            sizes.append(size)
            strides.append(stride)
    if len(sizes) > 3:
        return None
    sizes = [1] * (3 - len(sizes)) + sizes
    strides = [0] * (3 - len(strides)) + strides

    out = torch.empty(x.size(), dtype=dtype, device=x.device)
    num_rows = out.numel() // channels
    if num_rows == 0:
        return out
    block_rows = _BLOCK_SIZE // channels
    with torch.cuda.device(x.device):
        cast(Any, _rms_norm_kernel)[(triton.cdiv(num_rows, block_rows),)](
            x,
            weight if weight is not None else x,
            *(rope if rope is not None else (x, x)),
            out,
            num_rows,
            sizes[1],
            sizes[2],
            *strides,
            x.size(-3) if rope is not None else 1,
            x.size(-2) if rope is not None else 1,
            eps,
            channels=channels,
            block_rows=block_rows,
            has_weight=weight is not None,
            has_rope=rope is not None,
            num_warps=4,
        )
    return out
