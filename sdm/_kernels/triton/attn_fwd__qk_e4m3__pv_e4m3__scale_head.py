# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _attn_fwd__qk_e4m3__pv_e4m3__scale_head_kernel(
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


def attn_fwd__qk_e4m3__pv_e4m3__scale_head(
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
    r"""Attention forward pass with E4M3 QK and PV and per-head Q/K/V scales.

    The kernel follows the tiling of FlashAttention-2: each program computes
    one tile of ``query_tile`` query rows for one batch element and query
    head, and loops over tiles of ``context_tile`` key/value rows with an
    online softmax, so attention weights are never written to global memory.
    Both matrix multiplications take E4M3 inputs and accumulate in FP32:

    1. :math:`S = Q K^\top` from E4M3 queries and keys, rescaled by the
       query and key scales and ``scale``.
    2. Softmax over :math:`S` in FP32, with the weights :math:`P` multiplied
       by 256 and cast to E4M3.
    3. :math:`O = P V` from E4M3 weights and values, rescaled by the value
       scale and normalized in FP32.

    Queries, keys and values are quantized beforehand with one FP32 scale per
    batch element and head. Query heads share key/value heads when the number
    of query heads is a multiple of the number of key/value heads. Every
    context row is attended to; masks and padding are not supported.

    Args:
        q: Contiguous FP8 queries shaped ``[batch, query_heads, queries, D]``.
        k: Contiguous FP8 keys shaped ``[batch, kv_heads, context, D]``.
        v: Contiguous FP8 values shaped ``[batch, kv_heads, D, context]``.
        qs: Query scales shaped ``[batch, query_heads, 1, 1]``.
        ks: Key scales shaped ``[batch, kv_heads, 1, 1]``.
        vs: Value scales shaped ``[batch, kv_heads, 1, 1]``.
        dtype: Output dtype.
        query_tile: Query rows per program; a positive power of two.
        scale: Attention score multiplier. Defaults to ``1 / sqrt(D)``.
        context_tile: Context rows per loop iteration; a positive power of two.
        warps: Cooperating warps per program.
        stages: Software pipeline stages.
        accumulation_chunk: Tiles per FP32 accumulator flush; zero disables it.
        scale_weights_in_exp: Apply the weight factor 256 inside exp2 instead
            of before the weighted-value multiplication.
        fuse_score_scale: Permit fused score scaling and maximum subtraction.

    Returns:
        Attention output with the query shape and requested dtype.

    References:
        - `"FlashAttention-2: Faster Attention with Better Parallelism and Work
          Partitioning" <https://arxiv.org/abs/2307.08691>`_
        - `"Attention Quantization for Tabular Foundation Models"
          <https://arxiv.org/abs/2609.13031>`_
    """
    b, h, m, d = q.shape
    for name, size in (
        ("query_tile", query_tile),
        ("context_tile", context_tile),
        ("head dimension", d),
    ):
        if size <= 0 or size & (size - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if k.size(1) == 0 or h % k.size(1):
        raise ValueError("The number of K/V heads must divide the query heads")
    n = k.size(-2)
    # Q/K are [batch, heads, sequence, channels]; V is transposed.
    output = q.new_empty(q.shape, dtype=dtype)
    grid = (triton.cdiv(m, query_tile), b * h)  # ty: ignore[invalid-argument-type]
    with torch.cuda.device(q.device):
        cast(Any, _attn_fwd__qk_e4m3__pv_e4m3__scale_head_kernel)[grid](
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
