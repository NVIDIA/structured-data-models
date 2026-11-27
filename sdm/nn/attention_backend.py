# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor

from sdm._kernels.flash_attention import flash_attention
from sdm._kernels.sdpa import sdpa
from sdm.cache import QuantizedTensor

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
    r"""Base class for attention kernels of an attention layer."""

    @abc.abstractmethod
    def supports(
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        r"""Returns whether this backend can compute the attention call.

        Args:
            query: Queries with shape ``[B, Q, Hq, C]``.
            key: Keys with shape ``[B, KV, Hkv, C]``.
            value: Values with shape ``[B, KV, Hkv, Cv]``.
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
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:
        r"""Returns the attention output with shape ``[B, Q, Hq, Cv]``."""


class TorchAttentionBackend(AttentionBackend):
    r"""Attention backend using PyTorch's scaled dot-product attention."""

    def supports(  # noqa: D102
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return isinstance(key, Tensor) and isinstance(value, Tensor)

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key: Tensor | QuantizedTensor,  # [B, KV, Hkv, C]
        value: Tensor | QuantizedTensor,  # [B, KV, Hkv, Cv]
        *,
        attn_mask: Tensor | None,  # [B, Hq, Q, KV]
        seqused_key_value: Tensor | None,  # [B]
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, Cv]
        key, value = cast(Tensor, key), cast(Tensor, value)
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
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return (
            isinstance(key, Tensor)
            and isinstance(value, Tensor)
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
        )

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key: Tensor | QuantizedTensor,  # [B, KV, Hkv, C]
        value: Tensor | QuantizedTensor,  # [B, KV, Hkv, C]
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,  # [B]
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, C]
        return flash_attention(
            query=query,
            key=cast(Tensor, key),
            value=cast(Tensor, value),
            lengths=seqused_key_value,
            scale=scale,
        )


class FP8AttentionBackend(AttentionBackend):
    r"""Attention backend with FP8 (E4M3) matrix multiplications.

    It reads keys and values quantized with ``Float8PerHead`` and quantizes
    queries with one scale per batch element and head. It runs on GPUs with
    compute capability 8.9, 9.0, 12.0 or 12.1 and 32, 64, 128 or 256 channels
    per head, and declines masks, padding, causal attention, dropout,
    gradients, and ``torch.compile``.
    """

    def supports(  # noqa: D102
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return (
            _attn_fwd__qk_e4m3__pv_e4m3__scale_head is not None
            and not torch.compiler.is_compiling()
            and isinstance(key, QuantizedTensor)
            and isinstance(value, QuantizedTensor)
            and key.scale.size() == (*key.size()[:-3], 1, key.size(-2), 1)
            and value.scale.size()
            == (*value.size()[:-3], 1, value.size(-2), 1)
            and query.is_cuda
            and torch.cuda.get_device_capability(query.device)
            in {(8, 9), (9, 0), (12, 0), (12, 1)}
            and query.size(-1) in {32, 64, 128, 256}
            and value.size(-1) == query.size(-1)
            and attn_mask is None
            and seqused_key_value is None
            and not is_causal
            and dropout_p == 0.0
            and not (
                torch.is_grad_enabled()
                and (
                    query.requires_grad
                    or key.qdata.requires_grad
                    or key.scale.requires_grad
                    or value.qdata.requires_grad
                    or value.scale.requires_grad
                )
            )
        )

    def forward(  # noqa: D102
        self,
        query: Tensor,  # [B, Q, Hq, C]
        key: Tensor | QuantizedTensor,  # [B, KV, Hkv, C]
        value: Tensor | QuantizedTensor,  # [B, KV, Hkv, C]
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:  # [B, Q, Hq, C]
        assert _attn_fwd__qk_e4m3__pv_e4m3__scale_head is not None
        key, value = cast(QuantizedTensor, key), cast(QuantizedTensor, value)
        device = query.device.type
        dtype = (
            torch.get_autocast_dtype(device)
            if torch.is_autocast_enabled(device)
            else query.dtype
        )
        fp8_query = QuantizedTensor.quantize(query, dim=(-3, -1))

        # Launch settings measured to avoid register spills:
        channels, context = query.size(-1), key.size(-3)
        wide = channels == 256 or (channels == 128 and context >= 32768)
        blackwell = channels == 64 and torch.cuda.get_device_capability(
            query.device
        ) in {(12, 0), (12, 1)}

        out = _attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q=fp8_query.qdata.transpose(1, 2).contiguous(),  # [B, Hq, Q, C]
            k=key.qdata.transpose(1, 2).contiguous(),  # [B, Hkv, KV, C]
            v=value.qdata.permute(0, 2, 3, 1).contiguous(),  # [B, Hkv, C, KV]
            qs=fp8_query.scale.transpose(1, 2).contiguous(),  # [B, Hq, 1, 1]
            ks=key.scale.transpose(1, 2).contiguous(),  # [B, Hkv, 1, 1]
            vs=value.scale.transpose(1, 2).contiguous(),  # [B, Hkv, 1, 1]
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
