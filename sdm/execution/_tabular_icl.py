# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-owned native fit and one-global-layer cached replay."""

from collections import deque
from contextlib import nullcontext
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from sdm.execution._collectives import Exchange
from sdm.execution._tabular_cache import OwnedCache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.nn import SDPA, TransformerBlock


def fit_icl(
    model: ICLBlock,
    local: Tensor,
    labels: Tensor,
    cache: OwnedCache,
    exchange: Exchange,
    fit_head_gather_window: int | None = None,
) -> Tensor:
    """Preserve full projections/residuals; partition only SDPA heads."""
    full = exchange.gather(local, cache.lengths, 0)
    pending: deque[torch.cuda.Event] = deque()

    def attention(
        sdpa: SDPA,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if seqused_key_value is not None or attn_mask is not None or is_causal:
            raise ValueError("Context-sharded fit requires unmasked attention")
        if query.numel() == 0 or exchange.world == 1:
            return sdpa(query, key, value)
        if (
            fit_head_gather_window is not None
            and len(pending) >= fit_head_gather_window
        ):
            # Prior gathers on this rank are issued before another layer can
            # enqueue; peers reach the same calls in model order.
            pending.popleft().synchronize()
        scaled = exchange.attempt(
            lambda: (
                query
                if sdpa.query_scaling is None
                else sdpa.query_scaling(query, key_len=key.size(-3))
            )
        )
        q_heads, kv_heads = query.size(-2), key.size(-2)
        q_width, kv_width = (
            q_heads // exchange.world,
            kv_heads // exchange.world,
        )
        start = exchange.rank
        output = exchange.attempt(
            lambda: sdpa(
                scaled[..., start * q_width : (start + 1) * q_width, :],
                key[..., start * kv_width : (start + 1) * kv_width, :],
                value[..., start * kv_width : (start + 1) * kv_width, :],
                _already_scaled=True,
            )
        )
        source = exchange.attempt(lambda: output.movedim(-2, 0).contiguous())
        received = exchange.attempt(
            lambda: source.new_empty(
                (exchange.world * source.size(0), *source.shape[1:])
            )
        )
        dist.all_gather_into_tensor(received, source, group=exchange.group)
        result = exchange.attempt(lambda: received.movedim(0, -2).contiguous())
        if fit_head_gather_window is not None and result.is_cuda:
            # The copy of the gathered heads is ordered after NCCL on this
            # stream. Bound host enqueue lead without a device-wide fence.
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(result.device))
            pending.append(done)
        return result

    # Preserve the original FP32 policy; FP16 also permits native cuDNN.
    # Never enable quadratic math fit attention. Cached GQA prediction
    # deliberately retains native backend choice.
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    if (
        full.is_cuda
        and torch.is_autocast_enabled(full.device.type)
        and torch.get_autocast_dtype(full.device.type) == torch.float16
    ):
        backends.append(SDPBackend.CUDNN_ATTENTION)
    with sdpa_kernel(backends) if full.is_cuda else nullcontext():
        return model(full, labels, cache=cache, _attention=attention)


def workspace_bytes(model: ICLBlock, rows: int, cache: OwnedCache) -> int:
    """Conservative layer-local logical scratch, not total device memory."""
    context = sum(cache.lengths)
    # Actual cache dtype controls transport; preserve conservative FP32
    # attention scratch because native math attention can promote inputs.
    cache_element_size = next(cache._tensors()).element_size()
    largest = 0
    for module in model.layers:
        layer = cast(TransformerBlock, module)
        heads = layer.attn.num_query_heads
        kv_heads = model.kv_heads or layer.attn.num_key_value_heads
        width = layer.attn.head_dim
        score = heads * rows * context
        q = heads * rows * width
        kv = heads * context * width
        compact = kv_heads * context * 2 * width
        attention = (
            (13 * score + heads * rows + 8 * q + 12 * kv + 4 * compact + 8 * q)
            if rows
            else 0
        )
        gather = (
            (
                max(cache.lengths)
                + len(cache.lengths) * max(cache.lengths)
                + 2 * context
            )
            * kv_heads
            * 2
            * width
            * cache_element_size
            if len(cache.lengths) > 1
            else compact * cache_element_size
        )
        largest = max(largest, attention + gather)
    return largest


def predict_icl(model: ICLBlock, query: Tensor, cache: OwnedCache) -> Tensor:
    """Replay each full original query with one gathered KV layer at a time."""
    state = cache.exchange.attempt(query.clone)
    for index, layer in enumerate(model.layers):
        pair = cache[f"icl_block.layer{index}"]
        state = cache.exchange.attempt(
            lambda layer=layer, pair=pair, state=state: layer(
                query=state, key_value=pair, out=state
            )
        )
        del pair
    return cache.exchange.attempt(lambda: model.head(model.norm(state)))
