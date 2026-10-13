# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor

from sdm.cache import QuantizedKVCacheEntry


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    out_ptr,
    lse_ptr,
    n_splits: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kr: tl.constexpr,
    stride_kc: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vc: tl.constexpr,
    stride_vr: tl.constexpr,
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

    Each program scans its context partition in ``block_kv`` tiles for a
    ``[block_q, head_dim]`` query tile. Both matmuls use the full head width.

    """
    query_block, batch_query_head = tl.program_id(0), tl.program_id(1)
    batch = batch_query_head // n_q_heads
    query_head = batch_query_head % n_q_heads
    # Grouped-query attention shares K/V heads when n_q_heads > n_kv_heads.
    batch_kv_head = batch * n_kv_heads + query_head // (
        n_q_heads // n_kv_heads
    )
    kv_head = batch_kv_head % n_kv_heads
    split = tl.program_id(2) if n_splits > 1 else 0
    n_tiles = tl.cdiv(n_context, block_kv)  # ty: ignore[invalid-argument-type]
    first_tile = split * n_tiles // n_splits
    last_tile = (split + 1) * n_tiles // n_splits
    k_head_offset = batch * stride_kb + kv_head * stride_kh
    v_head_offset = batch * stride_vb + kv_head * stride_vh
    # Flatten packed heads directly; honor batch/head strides after slicing.
    if (
        stride_kb == n_kv_heads * n_context * head_dim
        and stride_kh == n_context * head_dim
    ):
        k_head_offset = batch_kv_head * n_context * head_dim
    if (
        stride_vb == n_kv_heads * n_context * head_dim
        and stride_vh == n_context * head_dim
    ):
        v_head_offset = batch_kv_head * n_context * head_dim
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
    for start in range(first_tile, last_tile):
        # 1. Tiled QK^T: reduce over channels to get [block_q, block_kv].
        indices = start * block_kv + cols
        k = tl.load(
            k_ptr
            + k_head_offset
            + indices[None, :] * stride_kr
            + dims[:, None] * stride_kc,
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
        v_offsets = (
            v_head_offset
            + indices[:, None] * stride_vr
            + dims[None, :] * stride_vc
        )
        v = tl.load(
            v_ptr + v_offsets,
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
            if ((start + 1 - first_tile) % acc_flush_tiles == 0) | (
                start + 1 == last_tile
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
    output_head = batch_query_head * n_splits + split
    if n_splits > 1:
        # Preserve each partition's softmax mass for the final reduction.
        logsum = row_max + tl.log2(weight_sum)
        if scale_weights_in_exp:
            logsum -= 8.0
        tl.store(
            lse_ptr + output_head * n_queries + rows,
            logsum,
            mask=rows < n_queries,
        )
    tl.store(
        out_ptr
        + output_head * n_queries * head_dim
        + rows[:, None] * head_dim
        + dims[None, :],
        out,
        mask=rows[:, None] < n_queries,
    )


@triton.jit
def _combine_attention_kernel(
    partial_ptr,
    lse_ptr,
    out_ptr,
    n_queries: tl.constexpr,
    head_dim: tl.constexpr,
    n_splits: tl.constexpr,
    block_splits: tl.constexpr,
) -> None:
    """Combine partition outputs, weighted by their softmax masses."""
    row, head = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, block_splits)
    dims = tl.arange(0, head_dim)
    offset = (head * n_splits + splits) * n_queries + row
    logsum = tl.load(
        lse_ptr + offset, mask=splits < n_splits, other=-float("inf")
    )
    weights = tl.exp2(logsum - tl.max(logsum, 0))  # ty: ignore[invalid-argument-type]
    partial = tl.load(
        partial_ptr + offset[:, None] * head_dim + dims[None, :],
        mask=splits[:, None] < n_splits,
        other=0.0,
    )
    out = tl.sum(partial * weights[:, None], 0) / tl.sum(weights, 0)  # ty: ignore[invalid-argument-type]
    tl.store(out_ptr + (head * n_queries + row) * head_dim + dims, out)


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
    output = x.new_empty(x.shape, dtype=torch.float8_e4m3fn)
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
            numel=length,
            block_size=block_size,
            num_heads=x.size(1),
            num_channels=x.size(-1),
            stride_b=x.stride(0),
            stride_h=x.stride(1),
            stride_r=x.stride(2),
            stride_c=x.stride(3),
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
    context_splits: int = 1,
) -> Tensor:
    """Run tiled attention on per-head scaled FP8 inputs.

    Args:
        q: Contiguous FP8 queries shaped ``[batch, query_heads, queries, D]``.
        k: FP8 keys shaped ``[batch, kv_heads, context, D]``.
        v: FP8 values shaped ``[batch, kv_heads, D, context]``.
            K/V may be strided, including views of selected cache heads.
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
        context_splits: Maximum number of context partitions per query tile.
            One scans the entire context; larger counts combine partial
            outputs using FP32 softmax masses.

    Returns:
        Attention output with the query shape and requested dtype.
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
    if context_splits < 1:
        raise ValueError("context_splits must be positive")
    n = k.size(-2)
    splits = min(context_splits, triton.cdiv(n, context_tile))  # ty: ignore[invalid-argument-type]
    # Q/K are [batch, heads, sequence, channels]; V is transposed.
    output = q.new_empty(q.shape, dtype=dtype)
    partial = output
    lse = None
    if splits > 1:
        partial = q.new_empty((b, h, splits, m, d), dtype=torch.float32)
        lse = q.new_empty((b, h, splits, m), dtype=torch.float32)
    grid = (triton.cdiv(m, query_tile), b * h, splits)  # ty: ignore[invalid-argument-type]
    with torch.cuda.device(q.device):
        cast(Any, _attention_kernel)[grid](
            q,
            k,
            v,
            qs,
            ks,
            vs,
            partial,
            lse,
            n_splits=splits,
            stride_kb=k.stride(0),
            stride_kh=k.stride(1),
            stride_kr=k.stride(2),
            stride_kc=k.stride(3),
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vc=v.stride(2),
            stride_vr=v.stride(3),
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
        if splits > 1:
            cast(Any, _combine_attention_kernel)[(m, b * h)](
                partial,
                lse,
                output,
                n_queries=m,
                head_dim=d,
                n_splits=splits,
                block_splits=triton.next_power_of_2(splits),  # ty: ignore[invalid-argument-type]
            )
    return output


def _fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cache: QuantizedKVCacheEntry | None = None,
    scale: float | None = None,
) -> tuple[Tensor, QuantizedKVCacheEntry]:
    """Quantize attention inputs and create or reuse their context cache."""
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query = query.to(dtype)
        if cache is None:
            key, value = key.to(dtype), value.to(dtype)
    batch_shape = torch.broadcast_shapes(
        query.shape[:-3], key.shape[:-3], value.shape[:-3]
    )

    def heads(x: Tensor) -> Tensor:
        return (
            x.expand(batch_shape + x.shape[-3:])
            .reshape(-1, *x.shape[-3:])
            .transpose(1, 2)
        )

    def rows(x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        return x.reshape(batch_shape + x.shape[-3:])

    q = heads(query)
    if cache is None:
        context = q[..., : key.size(-3), :]
        qs = context.abs().amax((-2, -1), keepdim=True).float()
        qs = qs.clamp_min(1e-12) / 448.0
        k, ks = quantize(heads(key))
        v, vs = quantize(heads(value))
        v = v.transpose(-1, -2).contiguous()
    else:
        k = heads(cache.key)
        v = heads(cache.value).transpose(-1, -2)
        ks, vs, qs = (
            heads(t).contiguous()
            for t in (cache.key_scale, cache.value_scale, cache.query_scale)
        )
    q, qs = quantize(q, qs)
    head_dim = query.size(-1)
    wide = head_dim == 256 or (head_dim == 128 and k.size(-2) >= 32768)
    # Measured launches avoid register spills; Ada retains accumulator flushes.
    blackwell = head_dim == 64 and torch.cuda.get_device_capability(
        query.device
    ) == (12, 0)
    query_tile = 64 if head_dim == 256 else 128
    context_splits = 1
    # Aligned V loads keep shorter predictions limited by launch overhead.
    min_context = (
        64 * 1024 if v.stride(-1) == 1 and v.stride(-2) % 8 == 0 else 32 * 1024
    )
    if cache is not None and k.size(-2) >= min_context:
        programs = triton.cdiv(q.size(-2), query_tile) * q.size(0) * q.size(1)  # ty: ignore[invalid-argument-type]
        sm_count = torch.cuda.get_device_properties(
            query.device
        ).multi_processor_count
        # Target four programs per SM, with at most 16 scratch partitions.
        context_splits = min(
            16,
            triton.next_power_of_2(triton.cdiv(4 * sm_count, programs)),  # ty: ignore[invalid-argument-type]
        )
    result = quantized_attention(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        dtype=query.dtype,
        query_tile=query_tile,
        scale=scale,
        context_tile=64 if wide or blackwell or k.size(-2) < 32768 else 128,
        warps=8 if wide else 4,
        stages=4 if blackwell else 2,
        accumulation_chunk=0 if blackwell else 16,
        scale_weights_in_exp=blackwell,
        fuse_score_scale=blackwell,
        context_splits=context_splits,
    )
    if cache is None:
        cache = QuantizedKVCacheEntry(
            key=rows(k),
            # Keep the logical cache shape while retaining the kernel layout.
            value=rows(v.transpose(-1, -2)),
            key_scale=rows(ks),
            value_scale=rows(vs),
            query_scale=rows(qs),
            dtype=query.dtype,
        )
    return rows(result), cache
