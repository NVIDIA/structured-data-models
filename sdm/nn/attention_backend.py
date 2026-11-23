# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc

import torch
from torch import Tensor

from sdm._kernels.flash_attention import flash_attention
from sdm._kernels.sdpa import sdpa
from sdm.nn import KeyValue


class AttentionBackend(abc.ABC):
    r"""Base class for attention kernels of an attention layer.

    A backend computes scaled dot-product attention on inputs with flattened
    batch dimensions, reading key/value projections in their storage format.
    """

    @abc.abstractmethod
    def supports(
        self,
        query: Tensor,
        key_value: KeyValue,
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
        key_value: KeyValue,
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
        key_value: KeyValue,
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
        key_value: KeyValue,  # [B, KV, Hkv, C], [B, KV, Hkv, Cv]
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
        key_value: KeyValue,
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
        key_value: KeyValue,  # [B, KV, Hkv, C]
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
