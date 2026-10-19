# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _load_input(
    x_ptr,
    cos_ptr,
    sin_ptr,
    row,
    input_offsets,
    channel_stride,
    c,
    mask,
    channels: tl.constexpr,
    seq_len,
    num_heads,
) -> tl.tensor:
    offsets = input_offsets[:, None]
    value = tl.load(
        x_ptr + offsets + c * channel_stride, mask=mask, other=0
    ).to(tl.float32)
    if cos_ptr is not None:
        first = c < channels // 2
        partner = tl.where(first, c + channels // 2, c - channels // 2)
        other = tl.load(
            x_ptr + offsets + partner * channel_stride, mask=mask, other=0
        ).to(tl.float32)
        table = (row // num_heads % seq_len)[:, None] * (channels // 2)
        table += c % (channels // 2)
        cos = tl.load(cos_ptr + table, mask=mask, other=0).to(tl.float32)
        sin = tl.load(sin_ptr + table, mask=mask, other=0).to(tl.float32)
        # Eager RoPE rounds both products and their sum in the input dtype.
        dtype = x_ptr.dtype.element_ty
        a = (value * cos).to(dtype).to(tl.float32)
        b = (other * sin).to(dtype).to(tl.float32)
        value = tl.where(first, a - b, a + b).to(dtype).to(tl.float32)
    return value


@triton.jit
def _variance(
    x_ptr,
    cos_ptr,
    sin_ptr,
    row,
    input_offsets,
    channel_stride,
    mask,
    channels: tl.constexpr,
    seq_len,
    num_heads,
    block_rows: tl.constexpr,
    strided: tl.constexpr,
) -> tl.tensor:
    groups = tl.arange(0, channels // 4)[None, :]
    total = tl.full((block_rows, channels // 4), 0, tl.float32)
    for i in tl.static_range(4):
        v = _load_input(  # ty: ignore[invalid-argument-type]
            x_ptr,
            cos_ptr,
            sin_ptr,
            row,
            input_offsets,
            channel_stride,
            groups * 4 + i,
            mask,
            channels,
            seq_len,
            num_heads,
        )
        total = tl.fma(v, v, total)
    if strided:
        # Preserve the contiguous kernel's within-warp reduction before
        # combining warp partials, regardless of the input load layout.
        for shift in tl.static_range(4, -1, -1):
            if (1 << shift) < channels // 4:
                indices = tl.broadcast_to(
                    groups ^ (1 << shift), (block_rows, channels // 4)
                )
                total += tl.gather(total, indices, axis=1)
        for shift in tl.static_range(6, 4, -1):
            if (1 << shift) < channels // 4:
                indices = tl.broadcast_to(
                    groups ^ (1 << shift), (block_rows, channels // 4)
                )
                total += tl.gather(total, indices, axis=1)
        total = tl.where(groups == 0, total, 0)
    elif channels == 32:
        # Preserve CUDA's shuffle order when Triton packs adjacent groups.
        for shift in tl.static_range(2, -1, -1):
            indices = tl.broadcast_to(groups ^ (1 << shift), (block_rows, 8))
            total += tl.gather(total, indices, axis=1)
        total = tl.where(groups == 0, total, 0)
    return tl.sum(total, 1)  # ty: ignore[invalid-argument-type]


@triton.jit
def _rmsnorm_cast(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    rows,
    seq_len,
    num_heads,
    row_sizes,
    row_strides,
    channel_stride,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    offsets = row[:, None] * channels
    if len(row_sizes) == 0:
        input_offsets = row * channels
        channel_stride = 1
    else:
        remaining = row
        input_offsets = tl.full((block_rows,), 0, tl.int64)
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
        x_ptr,
        cos_ptr,
        sin_ptr,
        row,
        input_offsets,
        channel_stride,
        mask,
        channels,
        seq_len,
        num_heads,
        block_rows,
        len(row_sizes) > 0,  # ty: ignore[invalid-argument-type]
    )
    c = tl.arange(0, channels)[None, :]
    value = _load_input(  # ty: ignore[invalid-argument-type]
        x_ptr,
        cos_ptr,
        sin_ptr,
        row,
        input_offsets,
        channel_stride,
        c,
        mask,
        channels,
        seq_len,
        num_heads,
    )
    value *= tl.rsqrt(variance[:, None] / channels + eps)
    if weight_ptr is not None:
        value *= tl.load(weight_ptr + c).to(tl.float32)
    tl.store(out_ptr + offsets + c, value, mask=mask)


@triton.jit(do_not_specialize=["rows", "seq_len", "num_heads"])
def _rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    rows,
    seq_len,
    num_heads,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    _rmsnorm_cast(  # ty: ignore[invalid-argument-type]
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
        seq_len,
        num_heads,
        (),
        (),
        1,
        channels,
        eps,
        block_rows,
    )


@triton.jit(
    do_not_specialize=[
        "rows",
        "seq_len",
        "num_heads",
        "row_sizes",
        "row_strides",
        "size0",
        "size1",
        "size2",
        "channel_stride",
    ]
)
def _strided_rmsnorm_cast_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    rows,
    seq_len,
    num_heads,
    row_sizes,
    row_strides,
    size0,
    size1,
    size2,
    channel_stride,
    channels: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
) -> None:
    if len(row_sizes) == 0:
        row_sizes = (size0, size1, size2)[: len(row_strides)]
    _rmsnorm_cast(  # ty: ignore[invalid-argument-type]
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows,
        seq_len,
        num_heads,
        row_sizes,
        row_strides,
        channel_stride,
        channels,
        eps,
        block_rows,
    )


def _row_size_arguments(
    sizes: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    # Triton specializes tuple elements even with do_not_specialize. Ordinary
    # attention layouts have at most three row axes; pass their sizes as
    # scalars so changing sequence length does not compile another kernel.
    if len(sizes) <= 3:
        size0, size1, size2 = (*sizes, 0, 0, 0)[:3]
        return (), (size0, size1, size2)
    return sizes, (0, 0, 0)


def rmsnorm_cast(
    x: Tensor,
    weight: Tensor | None,
    eps: float,
    rope: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    contiguous = x.is_contiguous()
    if weight is not None:
        weight = weight.contiguous()
    cos, sin = (
        (None, None)
        if rope is None
        else (table.contiguous() for table in rope)
    )
    channels = x.size(-1)
    rows = x.numel() // channels
    block_rows = 8
    grid = ((rows + block_rows - 1) // block_rows,)
    out = torch.empty_like(
        x,
        dtype=torch.get_autocast_dtype("cuda"),
        memory_format=torch.contiguous_format,
    )
    with torch.cuda.device(x.device):
        if contiguous:
            cast(Any, _rmsnorm_cast_kernel)[grid](
                x,
                weight,
                cos,
                sin,
                out,
                rows,
                x.size(-3) if rope is not None else 1,
                x.size(-2) if rope is not None else 1,
                channels,
                eps,
                block_rows,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            row_sizes, sizes = _row_size_arguments(x.shape[:-1] or (1,))
            cast(Any, _strided_rmsnorm_cast_kernel)[grid](
                x,
                weight,
                cos,
                sin,
                out,
                rows,
                x.size(-3) if rope is not None else 1,
                x.size(-2) if rope is not None else 1,
                row_sizes,
                x.stride()[:-1] or (0,),
                *sizes,
                x.stride(-1),
                channels,
                eps,
                block_rows,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
