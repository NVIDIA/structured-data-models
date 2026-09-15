# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import ModuleType

import torch
from torch import Tensor

from sdm.cache import QuantizedKVCacheEntry

_triton: ModuleType | None = None
try:
    from sdm._kernels.triton import fp8_attention as _triton_impl
except ImportError:
    pass
else:
    _triton = _triton_impl


def supports_fp8(query: Tensor) -> bool:
    return (
        _triton is not None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and query.is_cuda
        and (
            query.dtype in {torch.float16, torch.bfloat16}
            or (
                query.dtype == torch.float32
                and torch.is_autocast_enabled("cuda")
            )
        )
        and query.size(-1) in {32, 64, 128, 256}
        and torch.cuda.get_device_capability(query.device) in {(8, 9), (12, 0)}
    )


def fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cache: QuantizedKVCacheEntry | None = None,
    scale: float | None = None,
) -> tuple[Tensor, QuantizedKVCacheEntry]:
    assert _triton is not None
    # RMSNorm keeps Q/K in FP32. Match SDPA's autocast input conversion before
    # quantizing, without ever dequantizing cached FP8 keys and values.
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        query = query.to(dtype)
        if cache is None:
            key, value = key.to(dtype), value.to(dtype)
    batch_shape = torch.broadcast_shapes(
        query.shape[:-3], key.shape[:-3], value.shape[:-3]
    )

    def heads(x: Tensor) -> Tensor:
        return (
            x.expand(batch_shape + x.shape[-3:])
            .reshape(-1, *x.shape[-3:])
            .transpose(1, 2)
        )

    def rows(x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        return x.reshape(batch_shape + x.shape[-3:])

    q = heads(query)
    if cache is None:
        context = q[..., : key.size(-3), :]
        qs = context.abs().amax((-2, -1), keepdim=True).float()
        qs = qs.clamp_min(1e-12) / 448.0
        k, ks = _triton.quantize(heads(key))
        v, vs = _triton.quantize(heads(value))
    else:
        k = heads(cache.key).contiguous()
        v = heads(cache.value).contiguous()
        ks, vs, qs = (
            heads(t).contiguous()
            for t in (cache.key_scale, cache.value_scale, cache.query_scale)
        )
    q, qs = _triton.quantize(q, qs)
    blackwell = torch.cuda.get_device_capability(query.device) == (12, 0)
    wide = query.size(-1) == 256
    result = _triton.quantized_attention(
        q,
        k,
        v.transpose(-1, -2).contiguous(),
        qs,
        ks,
        vs,
        query.dtype,
        64 if wide else 128,
        scale,
        tile=64 if wide or blackwell or k.size(-2) < 32768 else 128,
        warps=8 if wide else 4,
        stages=4 if blackwell and not wide else 2,
        fused_accumulation=True,
        transposed_value=True,
        accumulation_chunk=0 if blackwell else 16,
        lift_exp=blackwell,
        fused_softmax=blackwell,
        context_splits=1,
    )
    if cache is None:
        cache = QuantizedKVCacheEntry(
            key=rows(k),
            value=rows(v),
            key_scale=rows(ks),
            value_scale=rows(vs),
            query_scale=rows(qs),
            dtype=query.dtype,
        )
    return rows(result), cache
