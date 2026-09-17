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


def supports_fp8(query: Tensor) -> bool:
    """Check whether the query and execution mode support the FP8 kernel.

    Args:
        query: Query tensor with channels in its final dimension.

    Returns:
        Whether Triton, hardware, head width, and inference mode are supported.
    """
    return (
        _triton_fp8_attention is not None
        and not torch.is_grad_enabled()
        and query.is_cuda
        and query.size(-1) in {32, 64, 128, 256}
        # CUDA compute capability (major, minor): Ada, Hopper, RTX Blackwell.
        # Restrict dispatch to architectures supported by this FP8 kernel.
        and torch.cuda.get_device_capability(query.device)
        in {(8, 9), (9, 0), (12, 0)}
    )


def fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cache: QuantizedKVCacheEntry | None = None,
    scale: float | None = None,
) -> tuple[Tensor, QuantizedKVCacheEntry]:
    """Compute FP8 attention and create or reuse a quantized context cache.

    Args:
        query: Queries shaped ``[..., rows, query_heads, channels]``. When
            creating a cache, the leading rows must include the context.
        key: Keys shaped ``[..., context_rows, kv_heads, channels]``.
        value: Values with the same shape as the keys.
        cache: Fitted quantized K/V and scales to reuse, or ``None`` to create
            them from the supplied context. Inputs must support FP8 inference.
        scale: Attention score multiplier; defaults to inverse square-root
            head width.

    Returns:
        The attention output and its quantized context cache.
    """
    assert _triton_fp8_attention is not None
    return _triton_fp8_attention(query, key, value, cache, scale)
