# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from sdm._kernels.flash_attention import flash_attention
from sdm._kernels.sdpa import sdpa
from sdm.cache import Float8PerHead, KVCacheEntry

_attn_fwd__qk_e4m3__pv_e4m3__scale_head: Callable[..., Tensor] | None = None
try:
    from sdm._kernels.triton.attn_fwd__qk_e4m3__pv_e4m3__scale_head import (
        attn_fwd__qk_e4m3__pv_e4m3__scale_head as _attn_fwd_impl,
    )
except ImportError:
    pass
else:
    _attn_fwd__qk_e4m3__pv_e4m3__scale_head = _attn_fwd_impl


class AttentionBackend(abc.ABC):
    r"""Base class for attention kernels of an attention layer.

    A backend computes scaled dot-product attention on inputs with flattened
    batch dimensions, reading key/value projections in their storage format.
    """

    @abc.abstractmethod
    def supports(
        self,
        query: Tensor,
        key_value: KVCacheEntry,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        r"""Returns whether this backend can compute the attention call.

        The result must depend on neither the batch size nor the number of
        queries, so that a layer uses the same kernel when recording and
        replaying its cache. Backends without gradients decline calls that
        need them.

        Args:
            query: Queries with shape ``[B, Q, Hq, C]``.
            key_value: Key/value projections with shapes ``[B, KV, Hkv, C]``
                and ``[B, KV, Hkv, Cv]``.
            attn_mask: Boolean or additive attention mask broadcastable to
                ``[B, Hq, Q, KV]``. ``None`` if ``seqused_key_value`` is given.
            seqused_key_value: Valid key/value lengths with shape ``[B]``.
            is_causal: Whether to apply an upper-left causal mask.
            dropout_p: Attention dropout probability.
            scale: Attention score multiplier. ``None`` uses
                ``1 / sqrt(C)``.
        """

    @abc.abstractmethod
    def forward(
        self,
        query: Tensor,
        key_value: KVCacheEntry,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:
        r"""Returns the attention output with shape ``[B, Q, Hq, Cv]``.

        It takes the arguments of :meth:`supports`.
        """


class TorchAttentionBackend(AttentionBackend):
    r"""Attention backend using PyTorch's scaled dot-product attention."""

    def supports(  # noqa: D102
        self,
        query: Tensor,
        key_value: KVCacheEntry,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return key_value.kv_cache_dtype is None

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key_value: KVCacheEntry,  # [B, KV, Hkv, C], [B, KV, Hkv, Cv]
        *,
        attn_mask: Tensor | None,  # [B, Hq, Q, KV]
        seqused_key_value: Tensor | None,  # [B]
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, Cv]
        key, value = key_value.key, key_value.value
        if seqused_key_value is not None:
            key_index = torch.arange(key.size(-3), device=key.device)
            attn_mask = key_index < seqused_key_value[:, None]  # [B, KV]
            attn_mask = attn_mask[:, None, None, :].expand(
                -1, -1, query.size(-3), -1
            )  # [B, 1, Q, KV]

        enable_gqa = False
        if query.size(-2) != key.size(-2):
            enable_gqa = True

        return sdpa(
            query=query.transpose(-3, -2),  # [B, Hq, Q, C]
            key=key.transpose(-3, -2),  # [B, Hkv, KV, C]
            value=value.transpose(-3, -2),  # [B, Hkv, KV, Cv]
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            scale=scale,
        )


class FlashAttention3Backend(AttentionBackend):
    r"""Attention backend using FlashAttention-3 on Hopper GPUs.

    It requires FP16/BF16 inputs and declines general masks, causal
    attention, and attention dropout.
    """

    def supports(  # noqa: D102
        self,
        query: Tensor,
        key_value: KVCacheEntry,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return (
            key_value.kv_cache_dtype is None
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
        )

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key_value: KVCacheEntry,  # [B, KV, Hkv, C]
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,  # [B]
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, C]
        return flash_attention(
            query=query,
            key=key_value.key,
            value=key_value.value,
            lengths=seqused_key_value,
            scale=scale,
        )


class FP8AttentionBackend(AttentionBackend):
    r"""Attention backend with FP8 (E4M3) matrix multiplications.

    It reads key/value projections stored as ``Float8PerHead`` and quantizes
    queries with one scale per batch element and head.
    """

    def supports(  # noqa: D102
        self,
        query: Tensor,
        key_value: KVCacheEntry,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
    ) -> bool:
        return (
            _attn_fwd__qk_e4m3__pv_e4m3__scale_head is not None
            and isinstance(key_value.kv_cache_dtype, Float8PerHead)
            and query.is_cuda
            and torch.cuda.get_device_capability(query.device)
            in {(8, 9), (9, 0), (12, 0), (12, 1)}
            and query.size(-1) in {32, 64, 128, 256}
            and attn_mask is None
            and seqused_key_value is None
            and not is_causal
            and dropout_p == 0.0
            and not (torch.is_grad_enabled() and query.requires_grad)
        )

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key_value: KVCacheEntry,  # [B, KV, Hkv, C]
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, C]
        assert _attn_fwd__qk_e4m3__pv_e4m3__scale_head is not None
        device = query.device.type
        dtype = (
            torch.get_autocast_dtype(device)
            if torch.is_autocast_enabled(device)
            else query.dtype
        )
        query, query_scale = Float8PerHead._quantize(query)  # [B, 1, Hq, 1]
        key_scale = cast(Tensor, key_value.key_scale)  # [B, 1, Hkv, 1]
        value_scale = cast(Tensor, key_value.value_scale)  # [B, 1, Hkv, 1]

        # Launch settings measured to avoid register spills:
        channels, context = query.size(-1), key_value.key.size(-3)
        wide = channels == 256 or (channels == 128 and context >= 32768)
        blackwell = channels == 64 and torch.cuda.get_device_capability(
            query.device
        ) == (12, 0)

        out = _attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            query.transpose(1, 2).contiguous(),  # [B, Hq, Q, C]
            key_value.key.transpose(1, 2).contiguous(),  # [B, Hkv, KV, C]
            key_value.value.permute(
                0, 2, 3, 1
            ).contiguous(),  # [B, Hkv, C, KV]
            query_scale.transpose(1, 2).contiguous(),  # [B, Hq, 1, 1]
            key_scale.transpose(1, 2).contiguous(),  # [B, Hkv, 1, 1]
            value_scale.transpose(1, 2).contiguous(),  # [B, Hkv, 1, 1]
            dtype=dtype,
            query_tile=64 if channels == 256 else 128,
            scale=scale,
            context_tile=64 if wide or blackwell or context < 32768 else 128,
            warps=8 if wide else 4,
            stages=4 if blackwell else 2,
            accumulation_chunk=0 if blackwell else 16,
            scale_weights_in_exp=blackwell,
            fuse_score_scale=blackwell,
        )  # [B, Hq, Q, C]
        return out.transpose(1, 2)
