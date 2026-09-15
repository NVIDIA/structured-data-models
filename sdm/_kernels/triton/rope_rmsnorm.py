import sys
from typing import Any, cast

import torch
from torch import Tensor

if sys.platform != "linux":
    raise ImportError("Triton kernels are only available on Linux")

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rope_rmsnorm_kernel(
    x_ptr,
    freq_ptr,
    out_ptr,
    length: tl.constexpr,
    heads: tl.constexpr,
    channels: tl.constexpr,
    rotary_channels: tl.constexpr,
    batch_stride: tl.constexpr,
    sequence_stride: tl.constexpr,
    head_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    interleaved: tl.constexpr,
    eps: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, channels)
    sequence = row // heads % length
    base = (
        row // (length * heads) * batch_stride
        + sequence * sequence_stride
        + row % heads * head_stride
    )
    freq_index = c // 2 if interleaved else c % (rotary_channels // 2)
    freq = tl.load(freq_ptr + freq_index, c < rotary_channels, 0)
    angle = sequence.to(tl.float32) * freq
    sin = libdevice.sin(angle).to(x_ptr.dtype.element_ty).to(tl.float32)
    cos = libdevice.cos(angle).to(x_ptr.dtype.element_ty).to(tl.float32)
    first = c % 2 == 0 if interleaved else c < rotary_channels // 2
    partner = (
        (c ^ 1)
        if interleaved
        else tl.where(
            first, c + rotary_channels // 2, c - rotary_channels // 2
        )
    )
    x = tl.load(x_ptr + base + c * channel_stride).to(tl.float32)
    y = tl.load(
        x_ptr + base + partner * channel_stride, c < rotary_channels, 0
    ).to(tl.float32)
    p = (x * cos).to(x_ptr.dtype.element_ty).to(tl.float32)
    q = (y * sin).to(x_ptr.dtype.element_ty).to(tl.float32)
    rotated = (
        tl.where(first, p - q, p + q).to(x_ptr.dtype.element_ty).to(tl.float32)
    )
    value = tl.where(c < rotary_channels, rotated, x)

    # Match native RMSNorm's four-channel accumulation and rounding order.
    groups = tl.arange(0, channels // 4)
    total = tl.full((channels // 4,), 0, tl.float32)
    for i in tl.static_range(4):
        v = tl.gather(value, groups * 4 + i, axis=0)
        total = tl.fma(v, v, total)
    variance = tl.sum(total, 0)  # ty: ignore[invalid-argument-type]
    inv = tl.rsqrt(variance / channels + eps)
    tl.store(out_ptr + row * channels + c, value * inv)


def rope_rmsnorm(
    x: Tensor,
    inv_freq: Tensor,
    *,
    rotary_channels: int,
    interleaved: bool,
    eps: float,
) -> Tensor:
    batch, length, heads, channels = x.shape
    dtype = torch.float32 if torch.is_autocast_enabled("cuda") else x.dtype
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    cast(Any, _rope_rmsnorm_kernel)[(batch * length * heads,)](
        x,
        inv_freq,
        out,
        length,
        heads,
        channels,
        rotary_channels,
        *x.stride(),
        interleaved,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
