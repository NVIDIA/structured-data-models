# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from sdm.cache import Cache, KVCacheEntry, QuantizedKVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock
from sdm.nn import Attention
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
def test_fp8_small_context_fallback(device: torch.device) -> None:
    module = Attention(128, 2, device=device)
    torch.nn.init.normal_(module.out_lin.weight, std=0.05)
    reference = copy.deepcopy(module)
    module.attention_quantization = "fp8"
    query = torch.randn(2, 19, 128, device=device)
    with torch.inference_mode():
        actual, cache = module(query, return_key_value=True)
        expected = reference(query)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert isinstance(cache, KVCacheEntry)
    assert module.state_dict().keys() == reference.state_dict().keys()


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_fp8_context_cache_and_chunking(dtype: torch.dtype) -> None:
    if torch.cuda.get_device_capability() not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 integration supports Ada, Hopper, and RTX Blackwell")
    module = TabICLv2TransformerBlock(
        channels=128,
        num_heads=2,
        norm_bias=True,
        qassmax=True,
        device="cuda",
    )
    torch.nn.init.normal_(module.attn.out_lin.weight, std=0.05)
    reference = copy.deepcopy(module)
    module.attn.attention_quantization = "fp8"
    context = torch.randn(2, 1, 8193, 128, device="cuda")
    query = torch.randn(2, 1, 129, 128, device="cuda")
    joined = torch.cat([context, query], dim=-2)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        expected = reference(joined, context)
        full = module(joined, context)
        _, cache = module(
            context, context, return_key_value=True, batch_size_limit=1
        )
        assert isinstance(cache, QuantizedKVCacheEntry)
        actual = module(query, cache, batch_size_limit=1)
        unchunked = module(query, cache)
        split = torch.cat(
            [
                module(query[..., :64, :], cache),
                module(query[..., 64:, :], cache),
            ],
            dim=-2,
        )
        empty = module(query[..., :0, :], cache)
        migrated = Cache(kv=cache).cpu().cuda()["kv"]
        replay = module(query, migrated)
    assert empty.size(-2) == 0
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, unchunked, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual, split, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual, replay, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        actual, full[..., -129:, :], atol=0.03, rtol=0.03
    )
    error = (full - expected).square().mean() / expected.square().mean()
    assert error.sqrt() < 0.03
    assert Cache(kv=cache).size() == sum(
        t.numel() * t.element_size() for t in cache._tensors()
    )
    assert cache.key.dtype == torch.float8_e4m3fn
    assert cache.value.dtype == torch.float8_e4m3fn
    assert cache.query_scale.dtype == torch.float32
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda", dtype=torch.float16, enabled=dtype == torch.float32
        ),
        pytest.raises(ValueError, match="dtype"),
    ):
        module(query, cache)
    with (
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
        pytest.raises(ValueError, match="inference"),
    ):
        module(query, cache)
