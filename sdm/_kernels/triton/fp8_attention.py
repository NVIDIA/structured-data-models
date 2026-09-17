# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as _tl

tl: Any = _tl


@triton.jit
def attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_queries: tl.constexpr,
    n_context: tl.constexpr,
    head_dim: tl.constexpr,
    attn_scale: tl.constexpr,
    block_q: tl.constexpr,
    block_kv: tl.constexpr,
    acc_flush_tiles: tl.constexpr = 0,
    scale_weights_in_exp: tl.constexpr = False,
    fuse_score_scale: tl.constexpr = False,
):
    """Write attention outputs for one query tile and head per program.

    Each program keeps a ``[block_q, head_dim]`` query tile and loops over
    context tiles of ``block_kv`` rows. Both matrix multiplications use
    the full head width.

    Args:
        q_ptr: FP8 queries shaped ``[batch, n_q_heads, n_queries, head_dim]``.
        k_ptr: FP8 keys shaped ``[batch, n_kv_heads, n_context, head_dim]``.
        v_ptr: FP8 values shaped ``[batch, n_kv_heads, head_dim, n_context]``.
        q_scale_ptr: Per-query-head dequantization scales.
        k_scale_ptr: Per-key-head dequantization scales.
        v_scale_ptr: Per-value-head dequantization scales.
        out_ptr: Output buffer with the query shape.
        n_q_heads: Query heads per batch item.
        n_kv_heads: K/V heads per batch item; must divide n_q_heads.
        n_queries: Number of query rows.
        n_context: Number of context rows.
        head_dim: Channels per head.
        attn_scale: Attention score multiplier.
        block_q: Query rows handled by each program.
        block_kv: Context rows processed per iteration.
        acc_flush_tiles: Tiles per FP32 accumulator flush; zero disables it.
        scale_weights_in_exp: Scale weights via ``exp2(x + 8)`` instead of
            multiplying ``exp2(x)`` by 256 afterward.
        fuse_score_scale: Move positive score scaling after the maximum
            reduction to permit fused scaling and maximum subtraction.
    """
    query_block, batch_query_head = tl.program_id(0), tl.program_id(1)
    batch = batch_query_head // n_q_heads
    query_head = batch_query_head % n_q_heads
    # Grouped-query attention shares K/V heads when n_q_heads > n_kv_heads.
    batch_kv_head = batch * n_kv_heads + query_head // (
        n_q_heads // n_kv_heads
    )
    rows = query_block * block_q + tl.arange(0, block_q)
    cols = tl.arange(0, block_kv)
    dims = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr
        + batch_query_head * n_queries * head_dim
        + rows[:, None] * head_dim
        + dims[None, :],
        rows[:, None] < n_queries,
        0.0,
    )
    # Restore Q/K magnitudes and convert natural-exponential scores to base 2.
    score_scale = (
        tl.load(q_scale_ptr + batch_query_head)
        * tl.load(k_scale_ptr + batch_kv_head)
        * attn_scale
        * 1.4426950408889634
    )
    value_scale = tl.load(v_scale_ptr + batch_kv_head)
    row_max = tl.full((block_q,), -float("inf"), tl.float32)
    weight_sum = tl.full((block_q,), 0, tl.float32)
    acc = tl.full((block_q, head_dim), 0, tl.float32)
    if acc_flush_tiles > 0:
        outer_acc = tl.full((block_q, head_dim), 0, tl.float32)
        outer_max = tl.full((block_q,), -float("inf"), tl.float32)
    for start in range(tl.cdiv(n_context, block_kv)):
        # 1. Tiled QK^T: reduce over channels to get [block_q, block_kv].
        indices = start * block_kv + cols
        k = tl.load(
            k_ptr
            + batch_kv_head * n_context * head_dim
            + indices[None, :] * head_dim
            + dims[:, None],
            indices[None, :] < n_context,
            0.0,
        )
        score = tl.dot(q, k, max_num_imprecise_acc=32)

        # 2. Online softmax: update row maxima and weight sums across tiles.
        if fuse_score_scale and attn_scale > 0:
            score = tl.where(
                indices[None, :] < n_context, score, -float("inf")
            )
            new_max = tl.maximum(row_max, tl.max(score, 1) * score_scale)
            shift = new_max - 8.0 if scale_weights_in_exp else new_max
            weights = tl.exp2(score * score_scale - shift[:, None])
        else:
            score = tl.where(
                indices[None, :] < n_context,
                score * score_scale,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(score, 1))
            if scale_weights_in_exp:
                weights = tl.exp2(score - (new_max[:, None] - 8.0))
            else:
                weights = tl.exp2(score - new_max[:, None])
        correction = tl.exp2(row_max - new_max)
        weight_sum = weight_sum * correction + tl.sum(weights, 1)
        acc *= correction[:, None]

        # 3. Tiled weights-V: reduce over context to get [block_q, head_dim].
        v_offsets = indices[:, None] + dims[None, :] * n_context
        v = tl.load(
            v_ptr + batch_kv_head * n_context * head_dim + v_offsets,
            indices[:, None] < n_context,
            0.0,
        )
        acc = tl.dot(
            (weights if scale_weights_in_exp else weights * 256.0).to(
                tl.float8e4nv
            ),
            v,
            acc,
            max_num_imprecise_acc=32,
        )
        # Periodically flush the dot accumulator to FP32 for long L4 contexts.
        if acc_flush_tiles > 0:  # noqa: SIM102 - constexpr guard avoids modulo zero.
            if ((start + 1) % acc_flush_tiles == 0) | (
                start + 1 == tl.cdiv(n_context, block_kv)
            ):
                outer_acc = (
                    outer_acc * tl.exp2(outer_max - new_max)[:, None] + acc
                )
                outer_max = new_max
                acc = tl.full((block_q, head_dim), 0, tl.float32)
        row_max = new_max
    if acc_flush_tiles > 0:
        acc = outer_acc
    # After all tiles, divide accumulated weighted values by the weight sum.
    out = acc * value_scale / weight_sum[:, None]
    if not scale_weights_in_exp:
        out *= 1.0 / 256.0
    tl.store(
        out_ptr
        + batch_query_head * n_queries * head_dim
        + rows[:, None] * head_dim
        + dims[None, :],
        out,
        rows[:, None] < n_queries,
    )


@triton.jit
def quantize_kernel(
    X,
    S,
    Y,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_r: tl.constexpr,
    stride_c: tl.constexpr,
):
    """Write scaled, clamped input values into a contiguous FP8 buffer.

    Args:
        X: Input shaped ``[batch, heads, rows, channels]``.
        S: Contiguous per-head dequantization scales.
        Y: Contiguous FP8 output buffer with the input shape.
        LENGTH: Number of elements per head.
        BLOCK: Elements handled by each program.
        H: Heads per batch item.
        C: Channels per head.
        stride_b: Input batch stride in elements.
        stride_h: Input head stride in elements.
        stride_r: Input row stride in elements.
        stride_c: Input channel stride in elements.
    """
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    head = tl.program_id(1)
    source = (
        (head // H) * stride_b
        + (head % H) * stride_h
        + (offset // C) * stride_r
        + (offset % C) * stride_c
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
        n_q_heads=h,
        n_kv_heads=k.size(1),
        n_queries=m,
        n_context=n,
        head_dim=d,
        attn_scale=d**-0.5 if scale is None else scale,
        block_q=query_tile,
        block_kv=context_tile,
        acc_flush_tiles=accumulation_chunk,
        scale_weights_in_exp=scale_weights_in_exp,
        fuse_score_scale=fuse_score_scale,
        num_warps=warps,
        num_stages=stages,
    )
    return output
