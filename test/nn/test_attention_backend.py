# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.cache import Float8PerHead, KVCacheEntry
from sdm.nn import SDPA
from sdm.nn.attention_backend import TorchAttentionBackend


@pytest.mark.parametrize("num_key_value_heads", [4, 2])
@pytest.mark.parametrize("mask", [None, "seqused_key_value", "attn_mask"])
def test_torch_attention_backend(num_key_value_heads: int, mask: str) -> None:
    query = torch.randn(3, 5, 4, 8)  # [B, Q, Hq, C]
    key = torch.randn(3, 7, num_key_value_heads, 8)  # [B, KV, Hkv, C]
    value = torch.randn_like(key)
    seqused_key_value = attn_mask = None
    if mask == "seqused_key_value":
        seqused_key_value = torch.tensor([7, 3, 1], dtype=torch.int32)
    elif mask == "attn_mask":
        attn_mask = torch.rand(3, 4, 5, 7) > 0.5  # [B, Hq, Q, KV]
        attn_mask[..., 0] = True

    expected = SDPA(
        num_query_heads=4, num_key_value_heads=num_key_value_heads
    )(query, key, value, seqused_key_value, attn_mask)

    backend = TorchAttentionBackend()
    key_value = KVCacheEntry(key=key, value=value)
    options = {
        "attn_mask": attn_mask,
        "seqused_key_value": seqused_key_value,
        "is_causal": False,
        "dropout_p": 0.0,
    }
    assert backend.supports(query, key_value, **options)
    out = backend.forward(query, key_value, **options, scale=None)
    torch.testing.assert_close(out, expected)

    quantized = Float8PerHead().quantize(key_value)
    assert not backend.supports(query, quantized, **options)
