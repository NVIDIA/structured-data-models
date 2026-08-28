"""Context-parallel attention over a sharded key/value axis.

For in-context models the context (key/value) length is the memory and compute
bottleneck. Sharding it across devices makes each device hold and process only
``1 / world_size`` of the context. Attention is a softmax-weighted reduction
over the key axis, so a shard split recombines exactly: each shard returns its
partial output together with the log-sum-exp normalizer of its scores, and an
online-softmax reduction over shards yields the same result as attending to the
full context (flash-/ring-attention math).

The reduction communicates only per-query state (``[..., Q, H, C]`` and
``[..., Q, H]``); its volume is independent of the context length.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

import torch
import torch.distributed as dist
from torch import Tensor

# The process group whose ranks hold the context shards, set by
# :func:`context_parallel` around a fit/predict call.
_context_group: ContextVar[dist.ProcessGroup | None] = ContextVar(
    "context_group", default=None
)
# The group over which :class:`~sdm.nn.SDPA` combines partial attention,
# activated by the model only around context-reducing attention.
_combine_group_var: ContextVar[dist.ProcessGroup | None] = ContextVar(
    "combine_group", default=None
)


@contextlib.contextmanager
def context_parallel(
    group: dist.ProcessGroup | None,
) -> Iterator[None]:
    r"""Run the enclosed fit/predict with the context sharded across ``group``.

    Each rank of ``group`` holds one shard of the in-context rows. In this
    scope, :meth:`~sdm.models.ICLModel.fit` stores only this rank's shard of
    the context cache, and :meth:`~sdm.models.ICLModel.predict` combines the
    sharded cross-attention across the group. ``None`` or a single-rank group
    runs exactly as the non-parallel code path.

    Args:
        group: The context-parallel process group, or ``None`` to disable.
    """
    token = _context_group.set(group)
    try:
        yield
    finally:
        _context_group.reset(token)


def context_group() -> dist.ProcessGroup | None:
    r"""Return the active context-parallel group, or ``None``."""
    return _context_group.get()


@contextlib.contextmanager
def combine_scope(group: dist.ProcessGroup | None) -> Iterator[None]:
    r"""Make :class:`~sdm.nn.SDPA` combine partial attention over ``group``.

    Scopes the cross-shard reduction to the context-reducing attention only
    (*e.g.* the dataset-wise block), leaving replicated attention untouched.

    Args:
        group: The group to combine over, or ``None`` to leave attention local.
    """
    token = _combine_group_var.set(group)
    try:
        yield
    finally:
        _combine_group_var.reset(token)


def _combine_group() -> dist.ProcessGroup | None:
    return _combine_group_var.get()


def shard_context(
    tensor: Tensor,
    group: dist.ProcessGroup | None,
    dim: int,
) -> Tensor:
    r"""Return this rank's contiguous shard of ``tensor`` along ``dim``.

    Splits ``tensor`` into ``get_world_size(group)`` near-equal parts (sizes
    differ by at most one) and returns this rank's part. A ``None`` or
    single-rank group returns ``tensor`` unchanged.

    Args:
        tensor: The tensor to shard.
        group: The context-parallel process group.
        dim: The dimension to shard along.

    Returns:
        This rank's contiguous shard.
    """
    if group is None or dist.get_world_size(group) == 1:
        return tensor
    rank = dist.get_rank(group)
    return tensor.tensor_split(dist.get_world_size(group), dim=dim)[
        rank
    ].contiguous()


def partial_attention(
    query: Tensor,  # [..., Q, H, C]
    key: Tensor,  # [..., KV, Hkv, C]
    value: Tensor,  # [..., KV, Hkv, C]
    *,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:  # [..., Q, H, C], [..., Q, H]
    r"""Attention over a local key/value shard with its log-sum-exp.

    Uses a memory-efficient attention kernel that returns the log-sum-exp of
    the scaled scores without materializing the ``[Q, KV]`` score matrix, so
    its memory is linear in the shard length. Grouped-/multi-query key and
    value heads are expanded to the query head count.

    Args:
        query: The query tensor with shape ``[..., Q, H, C]``. ``Q`` is the
            query length, ``H`` the number of query heads, ``C`` the channels
            per head.
        key: The local key shard with shape ``[..., KV, Hkv, C]``.
        value: The local value shard with shape ``[..., KV, Hkv, C]``.
        scale: Scaling factor for the scores. ``None`` uses ``1 / sqrt(C)``.

    Returns:
        The partial attention output with shape ``[..., Q, H, C]`` and the
        natural-log-scale log-sum-exp of the scores with shape ``[..., Q, H]``.
    """
    batch_shape = query.size()[:-3]
    q_len, num_query_heads, channels = query.size()[-3:]
    kv_len, num_kv_heads = key.size()[-3:-1]

    # Memory-efficient attention expects [B, H, S, C].
    q = query.transpose(-3, -2).reshape(-1, num_query_heads, q_len, channels)
    k = key.transpose(-3, -2).reshape(-1, num_kv_heads, kv_len, channels)
    v = value.transpose(-3, -2).reshape(-1, num_kv_heads, kv_len, channels)
    if num_kv_heads != num_query_heads:
        groups = num_query_heads // num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # `_scaled_dot_product_efficient_attention` returns the log-sum-exp that
    # `F.scaled_dot_product_attention` discards; it is the standard way to get
    # the normalizer needed to recombine shards. fp32/bf16, head_dim <= 256.
    out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, False, scale=scale
    )
    out = out.reshape(*batch_shape, num_query_heads, q_len, channels)
    out = out.transpose(-3, -2)  # [..., Q, H, C]
    lse = lse[..., :q_len].reshape(*batch_shape, num_query_heads, q_len)
    lse = lse.transpose(-2, -1)  # [..., Q, H]
    return out, lse


def combine_partials(
    partials: list[tuple[Tensor, Tensor]],
) -> Tensor:  # [..., Q, H, C]
    r"""Combine per-shard partial attention into the exact full attention.

    Each shard is weighted by ``softmax`` over its log-sum-exp normalizer,
    stabilized by the running maximum across shards. This is the in-process
    reduction; use :func:`combine` to reduce across a process group.

    Args:
        partials: The per-shard ``(output, log_sum_exp)`` pairs, with shapes
            ``[..., Q, H, C]`` and ``[..., Q, H]``.

    Returns:
        The combined attention output with shape ``[..., Q, H, C]``.
    """
    lse = torch.stack([shard_lse for _, shard_lse in partials], dim=0)
    running_max = lse.amax(dim=0)  # [..., Q, H]
    weight = (lse - running_max).exp()  # [G, ..., Q, H]
    denom = weight.sum(dim=0).unsqueeze(-1)  # [..., Q, H, 1]
    numer = sum(
        weight[i].unsqueeze(-1) * out for i, (out, _) in enumerate(partials)
    )
    return numer / denom


def combine(
    out: Tensor,  # [..., Q, H, C]
    lse: Tensor,  # [..., Q, H]
    group: dist.ProcessGroup | None = None,
) -> Tensor:  # [..., Q, H, C]
    r"""Combine a local partial attention across a context-parallel group.

    Performs the online-softmax reduction of :func:`combine_partials` over the
    ranks of ``group`` with a maximum and two sum all-reduces. The reduced
    output is identical on every rank and equal to attending to the full,
    unsharded context. A ``None`` or single-rank group is the identity.

    Args:
        out: This rank's partial attention output, shape ``[..., Q, H, C]``.
        lse: This rank's log-sum-exp with shape ``[..., Q, H]``.
        group: The process group whose ranks hold the context shards. ``None``
            uses no communication.

    Returns:
        The combined attention output with shape ``[..., Q, H, C]``.
    """
    world_size = 1 if group is None else dist.get_world_size(group)
    if world_size == 1:
        return out

    # Collectives require contiguous tensors; partial outputs are transposed
    # views, so materialize contiguous copies for each all-reduce.
    running_max = lse.clone(memory_format=torch.contiguous_format)
    dist.all_reduce(running_max, op=dist.ReduceOp.MAX, group=group)
    denom = (lse - running_max).exp().contiguous()  # [..., Q, H]
    numer = (out * denom.unsqueeze(-1)).contiguous()  # [..., Q, H, C]
    dist.all_reduce(numer, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=group)  # global denom
    return numer / denom.unsqueeze(-1)


def context_parallel_attention(
    query: Tensor,  # [..., Q, H, C]
    key: Tensor,  # [..., KV_local, Hkv, C]
    value: Tensor,  # [..., KV_local, Hkv, C]
    *,
    group: dist.ProcessGroup | None = None,
    scale: float | None = None,
) -> Tensor:  # [..., Q, H, C]
    r"""Exact attention whose key/value context is sharded across a group.

    Each rank passes its local key/value shard and the (replicated) query; the
    result equals attending to the concatenation of all shards. Length-aware
    query scaling, if any, must be applied by the caller using the global
    context length (see :func:`global_context_length`), since a rank sees only
    its shard length.

    Args:
        query: The replicated query with shape ``[..., Q, H, C]``.
        key: This rank's local key shard, shape ``[..., KV_local, Hkv, C]``.
        value: This rank's local value shard, same shape as ``key``.
        group: The context-parallel process group. ``None`` runs locally.
        scale: Scaling factor for the scores. ``None`` uses ``1 / sqrt(C)``.

    Returns:
        The combined attention output with shape ``[..., Q, H, C]``.
    """
    out, lse = partial_attention(query, key, value, scale=scale)
    return combine(out, lse, group=group)


def global_context_length(
    local_length: int,
    device: torch.device,
    group: dist.ProcessGroup | None = None,
) -> int:
    r"""Sum local context shard lengths across a context-parallel group.

    Length-aware query scaling (*e.g.* :class:`~sdm.nn.QASSMax`) must use the
    global context length rather than a single rank's shard length.

    Args:
        local_length: This rank's context shard length.
        device: The device used for the reduction tensor.
        group: The context-parallel process group. ``None`` returns
            ``local_length`` unchanged.

    Returns:
        The total context length summed over the group.
    """
    if group is None or dist.get_world_size(group) == 1:
        return local_length
    total = torch.tensor([local_length], device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return int(total.item())
