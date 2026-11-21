# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc

import torch
from torch import Tensor

from sdm._kernels.sdpa import sdpa
from sdm.cache import KVCacheEntry


class AttentionBackend(abc.ABC):
    r"""Base class for attention kernels of an attention layer.

    A backend computes scaled dot-product attention on inputs with flattened
    batch dimensions: queries with shape ``[B, Q, Hq, C]`` and key/value
    projections with shape ``[B, KV, Hkv, C]`` in their storage format.
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
    ) -> bool:
        r"""Returns whether this backend can compute the attention call.

        The result must not depend on the number of queries, so that a layer
        uses the same kernel when recording and replaying its cache.

        Args:
            query: Queries with shape ``[B, Q, Hq, C]``.
            key_value: Key/value projections with shape ``[B, KV, Hkv, C]``.
            attn_mask: Attention mask with shape ``[B, Hq, Q, KV]``.
            seqused_key_value: Valid key/value lengths with shape ``[B]``.
            is_causal: Whether to apply an upper-left causal mask.
            dropout_p: Attention dropout probability.
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
        r"""Returns the attention output with shape ``[B, Q, Hq, C]``.

        Args:
            query: Queries with shape ``[B, Q, Hq, C]``.
            key_value: Key/value projections with shape ``[B, KV, Hkv, C]``.
            attn_mask: Attention mask with shape ``[B, Hq, Q, KV]``.
            seqused_key_value: Valid key/value lengths with shape ``[B]``.
            is_causal: Whether to apply an upper-left causal mask.
            dropout_p: Attention dropout probability.
            scale: Attention score multiplier. ``None`` uses
                ``1 / sqrt(C)``.
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
    ) -> bool:
        return key_value.kv_cache_dtype is None

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key_value: KVCacheEntry,  # [B, KV, Hkv, C]
        *,
        attn_mask: Tensor | None,  # [B, Hq, Q, KV]
        seqused_key_value: Tensor | None,  # [B]
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, C]
        key, value = key_value.key, key_value.value
        if seqused_key_value is not None:
            key_index = torch.arange(key.size(-3), device=key.device)
            attn_mask = key_index < seqused_key_value[:, None]  # [B, KV]
            attn_mask = attn_mask[:, None, None, :].expand(
                -1, -1, query.size(-3), -1
            )  # [B, 1, Q, KV]

        return sdpa(
            query.transpose(-3, -2),  # [B, Hq, Q, C]
            key.transpose(-3, -2),  # [B, Hkv, KV, C]
            value.transpose(-3, -2),  # [B, Hkv, KV, C]
            attn_mask,
            dropout_p,
            query.size(-2) != key.size(-2),
            is_causal,
            scale,
        )
