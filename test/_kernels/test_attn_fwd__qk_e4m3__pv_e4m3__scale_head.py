# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
import torch
from torch import Tensor

from sdm.cache import Float8PerHead, KVCacheEntry

pytest.importorskip("triton")
from sdm._kernels.triton.attn_fwd__qk_e4m3__pv_e4m3__scale_head import (
    attn_fwd__qk_e4m3__pv_e4m3__scale_head,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() < (8, 9),
        reason="FP8 attention requires an Ada or newer CUDA GPU",
    ),
]


def quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    # [B, H, S, D] -> FP8 [B, H, S, D] and FP32 per-head scales [B, H, 1, 1].
    x = x.transpose(1, 2)
    entry = Float8PerHead().quantize(KVCacheEntry(key=x, value=x))
    scale = cast(Tensor, entry.key_scale)
    return (
        entry.key.transpose(1, 2).contiguous(),
        scale.transpose(1, 2).contiguous(),
    )


@pytest.mark.parametrize(
    ("batch", "queries", "context"), [(2, 129, 513), (1, 129, 8192)]
)
def test_fp8_partial_tiles(batch: int, queries: int, context: int) -> None:
    with torch.inference_mode():
        q = torch.randn(
            batch, 8, queries, 64, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            batch, 8, context, 64, device="cuda", dtype=torch.float16
        )
        v = torch.randn_like(k)
        q8, qs = quantize(q)
        k8, ks = quantize(k)
        v8, vs = quantize(v)
        actual = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q8,
            k8,
            v8.transpose(-1, -2).contiguous(),
            qs,
            ks,
            vs,
            q.dtype,
            128,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert actual.isfinite().all()
        relative_rms = (
            actual.float() - expected.float()
        ).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative_rms < 0.08


@pytest.mark.parametrize("scale", [None, 1.0])
@pytest.mark.parametrize("context_tile", [64, 128])
def test_grouped_query_attention_and_model_scale(
    scale: float | None, context_tile: int
) -> None:
    with torch.inference_mode():
        q = (
            torch.randn(2, 8, 129, 64, device="cuda", dtype=torch.float16)
            * 0.3
        )
        k = (
            torch.randn(2, 2, 8192, 64, device="cuda", dtype=torch.float16)
            * 0.3
        )
        v = torch.randn_like(k)
        q8, qs = quantize(q)
        k8, ks = quantize(k)
        v8, vs = quantize(v)
        actual = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q8,
            k8,
            v8.transpose(-1, -2).contiguous(),
            qs,
            ks,
            vs,
            q.dtype,
            128,
            scale,
            context_tile=context_tile,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, enable_gqa=True, scale=scale
        )
        assert actual.isfinite().all()
        relative_rms = (
            actual.float() - expected.float()
        ).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative_rms < 0.08


@pytest.mark.parametrize("accumulation_chunk", [0, 16])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("context_tile", [64, 128])
@pytest.mark.parametrize("context", [8193, 65536])
def test_periodic_attention_matches_cached_query_slice(
    context: int, context_tile: int, optimized: bool, accumulation_chunk: int
) -> None:
    with torch.inference_mode():
        q = torch.randn(1, 1, context, 64, device="cuda", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        q8, qs = quantize(q)
        k8, ks = quantize(k)
        v8, vs = quantize(v)
        v8 = v8.transpose(-1, -2).contiguous()
        opts: dict[str, Any] = {
            "accumulation_chunk": accumulation_chunk,
            "context_tile": context_tile,
            "scale_weights_in_exp": optimized,
            "fuse_score_scale": optimized,
        }
        full = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q8, k8, v8, qs, ks, vs, q.dtype, 128, **opts
        )
        cached = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q8[:, :, -129:].contiguous(),
            k8,
            v8,
            qs,
            ks,
            vs,
            q.dtype,
            128,
            **opts,
        )
        torch.testing.assert_close(
            cached, full[:, :, -129:], rtol=1e-3, atol=2e-5
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, -129:], k, v
        )
        relative_rms = (
            (cached.float() - reference.float()).square().mean()
            / reference.float().square().mean()
        ).sqrt()
        assert relative_rms < 0.08
        assert cached.isfinite().all()


def test_launch_over_grid_limit() -> None:
    # 8,193 batch elements and 8 heads exceed CUDA's limit of 65,535
    # programs, so the last 2 batch elements run in a second launch.
    with torch.inference_mode():
        q = torch.randn(8193, 8, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(8193, 8, 16, 64, device="cuda", dtype=torch.float16)
        q8, qs = quantize(q)
        k8, ks = quantize(k)
        v8, vs = quantize(torch.randn_like(k))
        v8 = v8.transpose(-1, -2).contiguous()
        out = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q=q8,
            k=k8,
            v=v8,
            qs=qs,
            ks=ks,
            vs=vs,
            dtype=q.dtype,
            query_tile=128,
        )
        for rows in (slice(None, 2), slice(-2, None)):
            expected = attn_fwd__qk_e4m3__pv_e4m3__scale_head(
                q=q8[rows],
                k=k8[rows],
                v=v8[rows],
                qs=qs[rows],
                ks=ks[rows],
                vs=vs[rows],
                dtype=q.dtype,
                query_tile=128,
            )
            torch.testing.assert_close(out[rows], expected, rtol=0, atol=0)


@pytest.mark.parametrize(("query_heads", "kv_heads"), [(3, 2), (1, 2), (2, 0)])
def test_invalid_attention_launch(query_heads: int, kv_heads: int) -> None:
    q = torch.empty(
        1, query_heads, 1, 64, device="cuda", dtype=torch.float8_e4m3fn
    )
    k = q.new_empty(1, kv_heads, 1, 64)
    qs = torch.ones(1, query_heads, 1, 1, device="cuda")
    ks = qs.new_ones(1, kv_heads, 1, 1)
    with pytest.raises(ValueError, match="K/V heads"):
        attn_fwd__qk_e4m3__pv_e4m3__scale_head(
            q,
            k,
            k.transpose(-1, -2).contiguous(),
            qs,
            ks,
            ks,
            dtype=torch.float16,
            query_tile=64,
        )
