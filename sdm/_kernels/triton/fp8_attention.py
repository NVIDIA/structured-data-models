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
def _attention_kernel(
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
    acc_flush_tiles: tl.constexpr,
    scale_weights_in_exp: tl.constexpr,
    fuse_score_scale: tl.constexpr,
) -> None:
    """Write attention outputs for one query tile and head per program.

    Each program keeps a ``[block_q, head_dim]`` query tile and loops over
    context tiles of ``block_kv`` rows. Both matrix multiplications use
    the full head width.

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
        mask=rows[:, None] < n_queries,
        other=0.0,
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
    for start in range(tl.cdiv(n_context, block_kv)):  # ty: ignore[invalid-argument-type]
        # 1. Tiled QK^T: reduce over channels to get [block_q, block_kv].
        indices = start * block_kv + cols
        k = tl.load(
            k_ptr
            + batch_kv_head * n_context * head_dim
            + indices[None, :] * head_dim
            + dims[:, None],
            mask=indices[None, :] < n_context,
            other=0.0,
        )
        score = tl.dot(q, k, max_num_imprecise_acc=32)

        # 2. Online softmax: update row maxima and weight sums across tiles.
        if fuse_score_scale and attn_scale > 0:
            score = tl.where(
                indices[None, :] < n_context, score, -float("inf")
            )
            new_max = tl.maximum(row_max, tl.max(score, 1) * score_scale)  # ty: ignore[invalid-argument-type]
            shift = new_max - 8.0 if scale_weights_in_exp else new_max
            weights = tl.exp2(score * score_scale - shift[:, None])
        else:
            score = tl.where(
                indices[None, :] < n_context,
                score * score_scale,
                -float("inf"),
            )
            new_max = tl.maximum(row_max, tl.max(score, 1))  # ty: ignore[invalid-argument-type]
            if scale_weights_in_exp:
                weights = tl.exp2(score - (new_max[:, None] - 8.0))
            else:
                weights = tl.exp2(score - new_max[:, None])
        correction = tl.exp2(row_max - new_max)
        weight_sum = weight_sum * correction + tl.sum(weights, 1)  # ty: ignore[invalid-argument-type]
        acc *= correction[:, None]

        # 3. Tiled weights-V: reduce over context to get [block_q, head_dim].
        v_offsets = indices[:, None] + dims[None, :] * n_context
        v = tl.load(
            v_ptr + batch_kv_head * n_context * head_dim + v_offsets,
            mask=indices[:, None] < n_context,
            other=0.0,
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
            if ((start + 1) % acc_flush_tiles == 0) | (  # ty: ignore[unsupported-operator]
                start + 1 == tl.cdiv(n_context, block_kv)  # ty: ignore[invalid-argument-type]
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
        mask=rows[:, None] < n_queries,
    )


@triton.jit
def _quantize_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    numel: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_channels: tl.constexpr,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_r: tl.constexpr,
    stride_c: tl.constexpr,
) -> None:
    """Write scaled, clamped input values into a contiguous FP8 buffer."""
    offset = tl.program_id(0) * block_size + tl.arange(0, block_size)
    head = tl.program_id(1)
    source = (
        (head // num_heads) * stride_b
        + (head % num_heads) * stride_h
        + (offset // num_channels) * stride_r
        + (offset % num_channels) * stride_c
    )
    x = tl.load(x_ptr + source, mask=offset < numel, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + head)
    y = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(out_ptr + head * numel + offset, y, mask=offset < numel)


def quantize(x: Tensor, scale: Tensor | None = None) -> tuple[Tensor, Tensor]:
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
    block_size = 1024
    grid = (
        triton.cdiv(length, block_size),  # ty: ignore[invalid-argument-type]
        x.size(0) * x.size(1),
    )
    with torch.cuda.device(x.device):
        cast(Any, _quantize_kernel)[grid](
            x,
            scale,
            output,
            length,
            block_size,
            x.size(1),
            x.size(-1),
            *x.stride(),
        )
    return output, scale


def quantized_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    qs: Tensor,
    ks: Tensor,
    vs: Tensor,
    dtype: torch.dtype,
    query_tile: int,
    scale: float | None = None,
    context_tile: int = 64,
    warps: int = 4,
    stages: int = 2,
    accumulation_chunk: int = 0,
    scale_weights_in_exp: bool = False,
    fuse_score_scale: bool = False,
) -> Tensor:
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
    grid = (triton.cdiv(m, query_tile), b * h)  # ty: ignore[invalid-argument-type]
    with torch.cuda.device(q.device):
        cast(Any, _attention_kernel)[grid](
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
