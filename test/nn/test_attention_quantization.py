# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import optimize
from sdm.cache import QuantizedKVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock
from sdm.nn import Attention
from sdm.testing import onlyCUDA

pytestmark = pytest.mark.usefixtures("fp8_rng")


@onlyCUDA
@pytest.mark.parametrize(("rows", "is_causal"), [(19, False), (8193, True)])
def test_fp8_attention_fallback(rows: int, is_causal: bool) -> None:
    module = Attention(64, 1, device="cuda")
    torch.nn.init.normal_(module.out_lin.weight, std=0.05)
    module._supports_quantized_attention = True
    query = torch.randn(1, rows, 64, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected = module(query, is_causal=is_causal)
        with optimize(attention="fp8"):
            actual, cache = module(
                query, is_causal=is_causal, return_key_value=True
            )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not isinstance(cache, QuantizedKVCacheEntry)


@onlyCUDA
def test_fp8_cache_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.version.cuda or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 requires CUDA compute capability 8.9 or newer")
    monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", "0.00001")
    block = TabICLv2TransformerBlock(
        channels=128, num_heads=2, norm_bias=True, qassmax=True, device="cuda"
    )
    torch.nn.init.normal_(block.attn.out_lin.weight, std=0.05)
    block.attn._supports_quantized_attention = True
    context = torch.randn(2, 1, 8193, 128, device="cuda")
    query = torch.randn(2, 1, 17, 128, device="cuda")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16),
        optimize(attention="fp8"),
    ):
        _, cache = block(
            context, context, return_key_value=True, batch_size_limit="auto"
        )
        assert isinstance(cache, QuantizedKVCacheEntry)
        expected = block(query, cache)
        actual = block(query, cache, batch_size_limit=1)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        assert block(query[..., :0, :], cache).size(-2) == 0
        with pytest.raises(ValueError, match="attention masks"):
            block.attn(query, cache, is_causal=True)
