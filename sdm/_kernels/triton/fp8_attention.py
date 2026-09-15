# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as _tl

tl: Any = _tl


@triton.jit
def attention_kernel(
    Q,
    K,
    V,
    QS,
    KS,
    VS,
    Out,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ACC_CHUNK: tl.constexpr = 0,
    LIFT_EXP: tl.constexpr = False,
    FUSED_SOFTMAX: tl.constexpr = False,
):
    block, head = tl.program_id(0), tl.program_id(1)
    kv_head = (head // HQ) * HK + (head % HQ) // (HQ // HK)
    rows = block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + head * M * D + rows[:, None] * D + dims[None, :],
        rows[:, None] < M,
        0.0,
    )
    scale = (
        tl.load(QS + head) * tl.load(KS + kv_head) * SCALE * 1.4426950408889634
    )
    vmax = tl.load(VS + kv_head)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    if ACC_CHUNK > 0:
        outer_acc = tl.full((BM, D), 0, tl.float32)
        outer_max = tl.full((BM,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(N, BN)):
        indices = start * BN + cols
        k = tl.load(
            K + kv_head * N * D + indices[None, :] * D + dims[:, None],
            indices[None, :] < N,
            0.0,
        )
        score = tl.dot(q, k, max_num_imprecise_acc=32)
        if FUSED_SOFTMAX and SCALE > 0:
            score = tl.where(indices[None, :] < N, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1) * scale)
            shift = new_max - 8.0 if LIFT_EXP else new_max
            p = tl.exp2(score * scale - shift[:, None])
        else:
            score = tl.where(
                indices[None, :] < N, score * scale, -float("inf")
            )
            new_max = tl.maximum(maximum, tl.max(score, 1))
            if LIFT_EXP:
                p = tl.exp2(score - (new_max[:, None] - 8.0))
            else:
                p = tl.exp2(score - new_max[:, None])
        correction = tl.exp2(maximum - new_max)
        denominator = denominator * correction + tl.sum(p, 1)
        acc *= correction[:, None]
        v_offsets = indices[:, None] + dims[None, :] * N
        v = tl.load(
            V + kv_head * N * D + v_offsets,
            indices[:, None] < N,
            0.0,
        )
        # Lift softmax probabilities by 256 to retain precision in FP8.
        acc = tl.dot(
            (p if LIFT_EXP else p * 256.0).to(tl.float8e4nv),
            v,
            acc,
            max_num_imprecise_acc=32,
        )
        # Periodically flush the dot accumulator to FP32 for long L4 contexts.
        if ACC_CHUNK > 0:  # noqa: SIM102 - constexpr guard avoids modulo zero.
            if ((start + 1) % ACC_CHUNK == 0) | (start + 1 == tl.cdiv(N, BN)):
                outer_acc = (
                    outer_acc * tl.exp2(outer_max - new_max)[:, None] + acc
                )
                outer_max = new_max
                acc = tl.full((BM, D), 0, tl.float32)
        maximum = new_max
    if ACC_CHUNK > 0:
        acc = outer_acc
    out = acc * vmax / denominator[:, None]
    if not LIFT_EXP:
        out *= 1.0 / 256.0
    tl.store(
        Out + head * M * D + rows[:, None] * D + dims[None, :],
        out,
        rows[:, None] < M,
    )


@triton.jit
def quantize_kernel(
    X,
    S,
    Y,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    source = (
        (head // H) * S0
        + (head % H) * S1
        + (offset // D) * S2
        + (offset % D) * S3
    )
    x = tl.load(X + source, offset < LENGTH, 0.0).to(tl.float32)
    scale = tl.load(S + head)
    y = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(Y + head * LENGTH + offset, y, offset < LENGTH)


def quantize(
    x: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = (
            x.abs().amax(dim=(-2, -1), keepdim=True).float().clamp_min(1e-12)
            / 448.0
        )
    output = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    length = x.size(-2) * x.size(-1)
    cast(Any, quantize_kernel)[
        (((length + 1023) // 1024), x.size(0) * x.size(1))
    ](x, scale, output, length, 1024, x.size(1), x.size(-1), *x.stride())
    return output, scale


def quantized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    dtype: torch.dtype,
    block: int,
    scale: float | None = None,
    tile: int = 64,
    warps: int = 4,
    stages: int = 2,
    accumulation_chunk: int = 0,
    lift_exp: bool = False,
    fused_softmax: bool = False,
) -> torch.Tensor:
    b, h, m, d = q.shape
    n = k.size(-2)
    # Q/K are [batch, heads, sequence, channels]; V is transposed.
    output = torch.empty((b, h, m, d), device=q.device, dtype=dtype)
    cast(Any, attention_kernel)[(((m + block - 1) // block), b * h)](
        q,
        k,
        v,
        qs,
        ks,
        vs,
        output,
        h,
        k.size(1),
        m,
        n,
        d,
        d**-0.5 if scale is None else scale,
        block,
        tile,
        accumulation_chunk,
        lift_exp,
        fused_softmax,
        num_warps=warps,
        num_stages=stages,
    )
    return output
