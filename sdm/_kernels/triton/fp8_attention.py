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
    """Write attention outputs for one query tile and head per program.

    Each program keeps a ``[BM, D]`` query tile and loops over context tiles
    of ``BN`` rows. Both matrix multiplications use the full head width D.

    Args:
        Q: FP8 queries shaped ``[batch, HQ, M, D]``.
        K: FP8 keys shaped ``[batch, HK, N, D]``.
        V: FP8 values shaped ``[batch, HK, D, N]``.
        QS: Per-query-head dequantization scales.
        KS: Per-key-head dequantization scales.
        VS: Per-value-head dequantization scales.
        Out: Output buffer shaped ``[batch, HQ, M, D]``.
        HQ: Query heads per batch item.
        HK: K/V heads per batch item; must divide HQ.
        M: Number of query rows.
        N: Number of context rows.
        D: Channels per head.
        SCALE: Attention score multiplier.
        BM: Query rows handled by each program.
        BN: Context rows processed per iteration.
        ACC_CHUNK: Tiles per FP32 accumulator flush; zero disables it.
        SCALE_WEIGHTS_IN_EXP: Scale weights via ``exp2(x + 8)`` instead of
            multiplying ``exp2(x)`` by 256 afterward.
        FUSE_SCORE_SCALE: Move positive score scaling after the maximum
            reduction to permit fused scaling and maximum subtraction.
    """
    query_block, batch_query_head = tl.program_id(0), tl.program_id(1)
    batch = batch_query_head // HQ
    query_head = batch_query_head % HQ
    # Query heads share K/V heads only when HQ > HK (grouped-query attention).
    batch_kv_head = batch * HK + query_head // (HQ // HK)
    rows = query_block * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    dims = tl.arange(0, D)
    q = tl.load(
        Q + batch_query_head * M * D + rows[:, None] * D + dims[None, :],
        rows[:, None] < M,
        0.0,
    )
    # Restore Q/K magnitudes and convert natural-exponential scores to base 2.
    score_scale = (
        tl.load(QS + batch_query_head)
        * tl.load(KS + batch_kv_head)
        * SCALE
        * 1.4426950408889634
    )
    value_scale = tl.load(VS + batch_kv_head)
    row_max = tl.full((BM,), -float("inf"), tl.float32)
    weight_sum = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    if ACC_CHUNK > 0:
        outer_acc = tl.full((BM, D), 0, tl.float32)
        outer_max = tl.full((BM,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(N, BN)):
        # 1. Tiled QK^T multiplication: [BM, D] @ [D, BN] -> [BM, BN].
        indices = start * BN + cols
        k = tl.load(
            K + batch_kv_head * N * D + indices[None, :] * D + dims[:, None],
            indices[None, :] < N,
            0.0,
        )
        score = tl.dot(q, k, max_num_imprecise_acc=32)

        # 2. Online softmax: update row maxima and weight sums across tiles.
        if FUSE_SCORE_SCALE and SCALE > 0:
            score = tl.where(indices[None, :] < N, score, -float("inf"))
            new_max = tl.maximum(row_max, tl.max(score, 1) * score_scale)
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
        correction = tl.exp2(row_max - new_max)
        weight_sum = weight_sum * correction + tl.sum(weights, 1)
        acc *= correction[:, None]

        # 3. Tiled weights-V multiplication: [BM, BN] @ [BN, D] -> [BM, D].
        v_offsets = indices[:, None] + dims[None, :] * N
        v = tl.load(
            V + batch_kv_head * N * D + v_offsets,
            indices[:, None] < N,
            0.0,
        )
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
    # After all tiles, divide accumulated weighted values by the weight sum.
    out = acc * value_scale / weight_sum[:, None]
    if not SCALE_WEIGHTS_IN_EXP:
        out *= 1.0 / 256.0
    tl.store(
        Out + batch_query_head * M * D + rows[:, None] * D + dims[None, :],
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
    """Write scaled, clamped input values into a contiguous FP8 buffer.

    Args:
        X: Input shaped ``[batch, heads, rows, channels]``.
        S: Contiguous per-head dequantization scales.
        Y: Contiguous FP8 output buffer with the input shape.
        LENGTH: Number of elements per head.
        BLOCK: Elements handled by each program.
        H: Heads per batch item.
        D: Channels per head.
        S0: Input batch stride in elements.
        S1: Input head stride in elements.
        S2: Input row stride in elements.
        S3: Input channel stride in elements.
    """
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    source = (
        (head // H) * S0
        + (head % H) * S1
        + (offset // D) * S2
        + (offset % D) * S3
    )
    x = tl.load(X + source, mask=offset < LENGTH, other=0.0).to(tl.float32)
    scale = tl.load(S + head)
    y = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(Y + head * LENGTH + offset, y, mask=offset < LENGTH)


def quantize(
    x: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 using per-head scales.

    Args:
        x: Input with shape ``[batch, heads, rows, channels]``.
        scale: Contiguous scales shaped ``[batch, heads, 1, 1]``. If omitted,
            derive scales from each head's maximum absolute input value.

    Returns:
        The contiguous FP8 tensor and its dequantization scales.
    """
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
    query_tile: int,
    scale: float | None = None,
    context_tile: int = 64,
    warps: int = 4,
    stages: int = 2,
    accumulation_chunk: int = 0,
    scale_weights_in_exp: bool = False,
    fuse_score_scale: bool = False,
) -> torch.Tensor:
    """Run tiled attention on per-head scaled FP8 inputs.

    Args:
        q: Contiguous FP8 queries shaped ``[batch, query_heads, queries, D]``.
        k: Contiguous FP8 keys shaped ``[batch, kv_heads, context, D]``.
        v: Contiguous FP8 values shaped ``[batch, kv_heads, D, context]``.
        qs: Query scales shaped ``[batch, query_heads, 1, 1]``.
        ks: Key scales shaped ``[batch, kv_heads, 1, 1]``.
        vs: Value scales shaped ``[batch, kv_heads, 1, 1]``.
        dtype: Output dtype.
        query_tile: Query rows handled by each program.
        scale: Attention score multiplier. Defaults to ``1 / sqrt(D)``.
        context_tile: Context rows processed per loop iteration.
        warps: Cooperating warps per program.
        stages: Software pipeline stages.
        accumulation_chunk: Tiles per FP32 accumulator flush; zero disables it.
        scale_weights_in_exp: Apply the weight factor 256 inside exp2 instead
            of before the weighted-value multiplication.
        fuse_score_scale: Permit fused score scaling and maximum subtraction.

    Returns:
        Attention output with the query shape and requested dtype.
    """
    b, h, m, d = q.shape
    n = k.size(-2)
    # Q/K are [batch, heads, sequence, channels]; V is transposed.
    output = torch.empty((b, h, m, d), device=q.device, dtype=dtype)
    cast(Any, attention_kernel)[(((m + query_tile - 1) // query_tile), b * h)](
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
        query_tile,
        context_tile,
        accumulation_chunk,
        scale_weights_in_exp,
        fuse_score_scale,
        num_warps=warps,
        num_stages=stages,
    )
    return output
