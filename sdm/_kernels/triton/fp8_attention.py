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
    SCALE_WEIGHTS_IN_EXP: tl.constexpr = False,
    FUSE_SCORE_SCALE: tl.constexpr = False,
):
    # Each program owns BM query rows for one head and visits every K/V tile.
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
    # Restore Q/K magnitudes and convert natural-exponential scores to base 2.
    score_scale = (
        tl.load(QS + head) * tl.load(KS + kv_head) * SCALE * 1.4426950408889634
    )
    value_scale = tl.load(VS + kv_head)
    row_max = tl.full((BM,), -float("inf"), tl.float32)
    weight_sum = tl.full((BM,), 0, tl.float32)
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
        # QK^T: [BM, D] @ [D, BN] -> [BM, BN].
        score = tl.dot(q, k, max_num_imprecise_acc=32)
        # Scale after max() so score * scale - shift can fuse.
        if FUSE_SCORE_SCALE and SCALE > 0:
            score = tl.where(indices[None, :] < N, score, -float("inf"))
            new_max = tl.maximum(row_max, tl.max(score, 1) * score_scale)
            # exp2(x + 8) = 256 * exp2(x): scale weights before FP8 conversion.
            shift = new_max - 8.0 if SCALE_WEIGHTS_IN_EXP else new_max
            weights = tl.exp2(score * score_scale - shift[:, None])
        else:
            score = tl.where(
                indices[None, :] < N, score * score_scale, -float("inf")
            )
            new_max = tl.maximum(row_max, tl.max(score, 1))
            if SCALE_WEIGHTS_IN_EXP:
                weights = tl.exp2(score - (new_max[:, None] - 8.0))
            else:
                weights = tl.exp2(score - new_max[:, None])
        # Rescale previous tiles when the running maximum increases.
        correction = tl.exp2(row_max - new_max)
        weight_sum = weight_sum * correction + tl.sum(weights, 1)
        acc *= correction[:, None]
        v_offsets = indices[:, None] + dims[None, :] * N
        v = tl.load(
            V + kv_head * N * D + v_offsets,
            indices[:, None] < N,
            0.0,
        )
        # Weighted V: scale unnormalized weights by 256 unless exp2 did it.
        acc = tl.dot(
            (weights if SCALE_WEIGHTS_IN_EXP else weights * 256.0).to(
                tl.float8e4nv
            ),
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
        row_max = new_max
    if ACC_CHUNK > 0:
        acc = outer_acc
    # Normalize and restore V's scale; an exp2 factor of 256 cancels here.
    out = acc * value_scale / weight_sum[:, None]
    if not SCALE_WEIGHTS_IN_EXP:
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
    scale_weights_in_exp: bool = False,
    fuse_score_scale: bool = False,
) -> torch.Tensor:
    """Run tiled attention on per-head scaled FP8 inputs.

    ``scale_weights_in_exp`` applies the weight factor 256 inside exp2;
    otherwise it is applied before the weighted-value multiplication.
    ``fuse_score_scale`` permits fused score scaling and max subtraction.
    Both select arithmetic variants, not different attention operations.
    """
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
        scale_weights_in_exp,
        fuse_score_scale,
        num_warps=warps,
        num_stages=stages,
    )
    return output
