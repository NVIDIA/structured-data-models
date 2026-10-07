# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental inference context parallelism for cached ICL attention.

Every rank fits the same context and predicts the same query batches in the
same order. Only the saved ICL key/value sequence is partitioned. Fit compute,
model parameters, row embedding, and relational message passing are replicated.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor

from sdm.cache import Cache, KVCacheEntry

_group: ContextVar[dist.ProcessGroup | None] = ContextVar(
    "context_group", default=None
)
_attention: ContextVar[tuple[dist.ProcessGroup, int] | None] = ContextVar(
    "context_attention", default=None
)
_kernel: ContextVar[Literal["efficient", "flash"]] = ContextVar(
    "context_kernel", default="efficient"
)


@contextmanager
def context_parallel(
    group: dist.ProcessGroup,
    *,
    kernel: Literal["efficient", "flash"] = "efficient",
) -> Iterator[None]:
    """Shard fitted ICL caches and combine predictions across ``group``.

    All ranks must enter with identical model weights, preprocessing state,
    contexts, queries, and batching. Gradients and compiled execution are not
    supported. Keep this scope active during both fit and predict.
    ``kernel="flash"`` uses native GQA without expanding KV heads and requires
    CUDA FP16/BF16. The default efficient kernel also supports FP32.
    """
    if torch.is_grad_enabled():
        raise RuntimeError(
            "Context parallelism requires inference_mode or no_grad"
        )
    token = _group.set(group)
    kernel_token = _kernel.set(kernel)
    try:
        yield
    finally:
        _group.reset(token)
        _kernel.reset(kernel_token)


def shard_cached_context(
    kv: KVCacheEntry, cache: Cache, key: str
) -> KVCacheEntry:
    """Store topology and global length, then copy this rank's KV shard."""
    group = _group.get()
    if group is None:
        return kv
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    length = kv.key.size(-3)
    cache[key + ".context_parallel"] = (world, rank, length)
    # Clone releases the full fit storage retained by a contiguous slice.
    return KVCacheEntry(
        *(tensor.tensor_split(world, dim=-3)[rank].clone() for tensor in kv)
    )


@contextmanager
def cached_context(cache: Cache | None, key: str) -> Iterator[None]:
    """Activate distributed attention only when replaying a sharded cache."""
    group = _group.get()
    metadata = None if cache is None else cache.get(key + ".context_parallel")
    if cache is None and group is not None:
        raise RuntimeError(
            "Context parallelism requires separate fit and predict"
        )
    state = None
    if metadata is not None:
        world, rank, length = cast(tuple[int, int, int], metadata)
        if group is None or (world, rank) != (
            dist.get_world_size(group),
            dist.get_rank(group),
        ):
            raise RuntimeError(
                "Sharded cache requires its original rank topology"
            )
        state = (group, length)
    elif cache is not None and cache.is_replaying and group is not None:
        raise RuntimeError(
            "Fit the cache inside context_parallel before prediction"
        )
    token = _attention.set(state)
    try:
        yield
    finally:
        _attention.reset(token)


def attention_context() -> tuple[dist.ProcessGroup, int] | None:
    """Return the current replay group and global context length."""
    return _attention.get()


def partial_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    kernel: Literal["efficient", "flash"] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return local attention and natural-log normalizer in [..., Q, H, C].

    Uses the PyTorch efficient CUDA kernel and a dense CPU reference. Supports
    arbitrary broadcast batch dimensions, GQA, and empty shards. Masks, causal
    attention, and variable valid lengths are deliberately not supported.
    """
    if query.is_cuda and torch.is_autocast_enabled("cuda"):
        # Public SDPA is autocast-registered; the private LSE operator is not.
        # RMSNorm can promote Q to FP32 while the fitted KV remains BF16.
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = (
            tensor.to(dtype) if tensor.dtype != torch.float64 else tensor
            for tensor in (query, key, value)
        )
    kernel = _kernel.get() if kernel is None else kernel
    if kernel == "flash" and (
        not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError("Flash context attention requires CUDA FP16 or BF16")
    batch = torch.broadcast_shapes(
        query.shape[:-3], key.shape[:-3], value.shape[:-3]
    )
    qlen, heads, channels = query.shape[-3:]
    klen, kv_heads, _ = key.shape[-3:]
    out_shape = (*batch, qlen, heads, value.size(-1))
    if klen == 0 or qlen == 0 or 0 in batch:
        return query.new_zeros(out_shape), torch.full(
            out_shape[:-1],
            -torch.inf,
            device=query.device,
            dtype=torch.float64
            if query.dtype == torch.float64
            else torch.float32,
        )
    q = (
        query.expand(*batch, *query.shape[-3:])
        .reshape(-1, qlen, heads, channels)
        .transpose(1, 2)
    )
    k = (
        key.expand(*batch, *key.shape[-3:])
        .reshape(-1, klen, kv_heads, channels)
        .transpose(1, 2)
    )
    v = (
        value.expand(*batch, *value.shape[-3:])
        .reshape(-1, klen, kv_heads, value.size(-1))
        .transpose(1, 2)
    )
    if heads != kv_heads and kernel != "flash":
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
    if query.is_cuda and kernel == "flash":
        out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, False, False, scale=scale
        )
        lse = lse[..., :qlen]
    elif query.is_cuda:
        out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, None, True, 0.0, False, scale=scale
        )
        lse = lse[..., :qlen]
    else:
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        scores = q.to(dtype) @ k.to(dtype).transpose(-1, -2)
        scores *= channels**-0.5 if scale is None else scale
        lse = scores.logsumexp(-1)
        out = (scores.softmax(-1) @ v.to(dtype)).to(query.dtype)
    return (
        out.transpose(1, 2).reshape(out_shape),
        lse.transpose(1, 2).reshape(out_shape[:-1]),
    )


def context_parallel_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    group: dist.ProcessGroup,
    scale: float | None = None,
    kernel: Literal["efficient", "flash"] | None = None,
) -> Tensor:
    """Combine local KV attention using stable FP32 all-reductions.

    The query must be identical on all ranks. Empty global contexts return
    zero. Floating-point reductions need tolerance-based, not bitwise, parity.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("Context parallel attention is inference-only")
    out, lse = partial_attention(query, key, value, scale=scale, kernel=kernel)
    if dist.get_world_size(group) == 1 or out.numel() == 0:
        return out
    maximum = lse.contiguous().clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    weight = (lse - maximum.nan_to_num(neginf=0)).exp()
    # Pack numerator and denominator into one SUM collective after MAX.
    dtype = torch.float64 if out.dtype == torch.float64 else torch.float32
    weighted = out.to(dtype) * weight.unsqueeze(-1)
    combined = torch.cat((weighted, weight.unsqueeze(-1)), dim=-1).contiguous()
    dist.all_reduce(combined, group=group)
    numerator, denominator = combined[..., :-1], combined[..., -1:]
    return (numerator / denominator.clamp_min(torch.finfo(dtype).tiny)).to(
        out.dtype
    )
