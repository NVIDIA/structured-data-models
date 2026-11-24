# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import Tensor

_flash_attn_varlen_func: Callable[..., Tensor] | None = None


def load_flash_attention() -> None:
    """Import FlashAttention-3 once, outside any compiled region.

    ``torch.compile`` then traces the already bound function, whose
    FlashAttention-3 3.0 kernels are registered ``torch.library`` operators
    with fake implementations, instead of an import statement.
    """
    global _flash_attn_varlen_func
    from flash_attn_interface import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
        flash_attn_varlen_func,
    )

    _flash_attn_varlen_func = flash_attn_varlen_func


def flash_attention(
    query: Tensor,  # [B, Q, Hq, D]
    key: Tensor,  # [B, K, Hkv, D]
    value: Tensor,  # [B, K, Hkv, D]
    *,
    lengths: Tensor | None,
    scale: float | None,
) -> Tensor:
    if _flash_attn_varlen_func is None:
        load_flash_attention()
    flash_attn_varlen_func = _flash_attn_varlen_func
    assert flash_attn_varlen_func is not None

    # Match SDPA's autocast policy after FP32 query/key normalization.
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)

    # FA3's backward preprocessing uses CUDA grid.z for the batch axis.
    outputs = []
    for start in range(0, query.size(0), 65_535):
        q = query[start : start + 65_535]
        k = key[start : start + 65_535]
        v = value[start : start + 65_535]
        used = (
            None
            if lengths is None
            else lengths[start : start + 65_535].contiguous()
        )
        if used is not None:
            # Masking zeroes the gradients of padded keys and values.
            valid = torch.arange(k.size(1), device=k.device) < used[:, None]
            k = k.masked_fill(~valid[..., None, None], 0)
            v = v.masked_fill(~valid[..., None, None], 0)
        cu = torch.arange(q.size(0) + 1, dtype=torch.int32, device=q.device)
        out = flash_attn_varlen_func(
            q=q.flatten(0, 1).contiguous(),
            k=k.flatten(0, 1).contiguous(),
            v=v.flatten(0, 1).contiguous(),
            cu_seqlens_q=cu * q.size(1),
            cu_seqlens_k=cu * k.size(1),
            seqused_k=used,
            max_seqlen_q=q.size(1),
            max_seqlen_k=k.size(1),
            softmax_scale=scale,
        )
        out = out[0] if isinstance(out, tuple) else out
        outputs.append(out.view(*q.shape[:-1], value.size(-1)))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
