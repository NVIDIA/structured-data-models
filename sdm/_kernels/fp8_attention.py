# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import Tensor

from sdm.cache import QuantizedKVCacheEntry

_FP8Attention: TypeAlias = Callable[
    [Tensor, Tensor, Tensor, QuantizedKVCacheEntry | None, float | None],
    tuple[Tensor, QuantizedKVCacheEntry],
]

_triton_fp8_attention: _FP8Attention | None = None
try:
    from sdm._kernels.triton.fp8_attention import (
        _fp8_attention as _triton_fp8_attention_impl,
    )
except ImportError:
    pass
else:
    _triton_fp8_attention = _triton_fp8_attention_impl


def fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cache: QuantizedKVCacheEntry | None = None,
    scale: float | None = None,
) -> tuple[Tensor, QuantizedKVCacheEntry] | None:
    """Compute FP8 attention and create or reuse a quantized context cache.

    Args:
        query: Queries shaped ``[..., rows, query_heads, channels]``. When
            creating a cache, the leading rows must include the context.
        key: Keys shaped ``[..., context_rows, kv_heads, channels]``.
        value: Values with the same shape as the keys.
        cache: Fitted quantized K/V and scales to reuse, or ``None`` to create
            them from the supplied context.
        scale: Attention score multiplier; defaults to inverse square-root
            head width.

    Returns:
        The attention output and its quantized context cache, or ``None``
        when unsupported inputs require the caller's regular attention path.

    Raises:
        ValueError: A fitted quantized cache cannot be used with the current
            device, head width, backend, or execution mode.
    """
    if (
        _triton_fp8_attention is not None
        and not torch.is_grad_enabled()
        and query.is_cuda
        and query.size(-1) in {32, 64, 128, 256}
        # CUDA compute capability (major, minor): Ada, Hopper, RTX Blackwell.
        # Restrict dispatch to architectures supported by this FP8 kernel.
        and torch.cuda.get_device_capability(query.device)
        in {(8, 9), (9, 0), (12, 0)}
    ):
        if cache is not None and query.numel() == 0:
            return query, cache
        return _triton_fp8_attention(query, key, value, cache, scale)
    if cache is not None:
        raise ValueError(
            "Fitted FP8 attention requires supported CUDA inference"
        )
    return None
