# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

pytest.importorskip("triton")
from sdm._kernels.triton.fp8_attention import (
    quantize,
    quantized_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 9),
    reason="FP8 attention requires an Ada or newer CUDA GPU",
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
        actual = quantized_attention(
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
@pytest.mark.parametrize("tile", [64, 128])
def test_grouped_heads_and_model_scale(scale: float | None, tile: int) -> None:
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
        actual = quantized_attention(
            q8,
            k8,
            v8.transpose(-1, -2).contiguous(),
            qs,
            ks,
            vs,
            q.dtype,
            128,
            scale,
            tile=tile,
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
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("context", [8193, 65536])
def test_periodic_attention_matches_cached_query_slice(
    context: int, tile: int, optimized: bool, accumulation_chunk: int
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
            "tile": tile,
            "scale_weights_in_exp": optimized,
            "fuse_score_scale": optimized,
        }
        full = quantized_attention(
            q8, k8, v8, qs, ks, vs, q.dtype, 128, **opts
        )
        cached = quantized_attention(
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


def test_quantization_reads_strided_heads_without_fp16_copy() -> None:
    x = torch.randn(
        2, 513, 8, 64, device="cuda", dtype=torch.float16
    ).transpose(1, 2)
    actual, actual_scale = quantize(x)
    expected, expected_scale = quantize(x.contiguous())
    assert actual.is_contiguous()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
