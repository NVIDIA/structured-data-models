# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn import SDPA, Float8PerHead, KeyValue
from sdm.nn.attention_backend import AttentionBackend, TorchAttentionBackend


@pytest.mark.parametrize("num_key_value_heads", [4, 2])
@pytest.mark.parametrize("mask", [None, "seqused_key_value", "attn_mask"])
def test_torch_attention_backend(num_key_value_heads: int, mask: str) -> None:
    query = torch.randn(3, 5, 4, 8)  # [B, Q, Hq, C]
    key = torch.randn(3, 7, num_key_value_heads, 8)  # [B, KV, Hkv, C]
    value = torch.randn_like(key)
    seqused_key_value = attn_mask = expected_mask = None
    if mask == "seqused_key_value":
        seqused_key_value = torch.tensor([7, 3, 1], dtype=torch.int32)
        key_index = torch.arange(7)
        expected_mask = key_index < seqused_key_value[:, None, None, None]
    elif mask == "attn_mask":
        attn_mask = expected_mask = torch.rand(3, 4, 5, 7) > 0.5
        attn_mask[..., 0] = True

    expected = F.scaled_dot_product_attention(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=expected_mask,
        scale=0.5,
        enable_gqa=True,
    ).transpose(1, 2)

    backend = TorchAttentionBackend()
    key_value = KeyValue(key=key, value=value)
    options = {
        "attn_mask": attn_mask,
        "seqused_key_value": seqused_key_value,
        "is_causal": False,
        "dropout_p": 0.0,
        "scale": 0.5,
    }
    assert backend.supports(query, key_value, **options)
    out = backend.forward(query, key_value, **options)
    torch.testing.assert_close(out, expected)

    quantized = Float8PerHead().quantize(key_value)
    assert not backend.supports(query, quantized, **options)


class _ZeroBackend(AttentionBackend):
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def supports(
        self,
        query: Tensor,
        key_value: KeyValue,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return self.supported and key_value.kv_cache_dtype is None

    def forward(
        self,
        query: Tensor,
        key_value: KeyValue,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> Tensor:
        return torch.zeros_like(query)


def test_sdpa_backend() -> None:
    query = torch.randn(2, 3, 5, 4, 8)  # [..., Q, Hq, C]
    key = torch.randn(2, 3, 7, 4, 8)  # [..., KV, Hkv, C]
    key_value = KeyValue(key=key, value=torch.randn_like(key))
    torch_sdpa = SDPA(num_query_heads=4)
    supported = SDPA(num_query_heads=4, backend=_ZeroBackend(True))
    unsupported = SDPA(num_query_heads=4, backend=_ZeroBackend(False))

    assert supported(query, key_value).eq(0).all()
    expected = torch_sdpa(query, key_value)
    torch.testing.assert_close(unsupported(query, key_value), expected)

    # Backends that can't read quantized key/value projections get them
    # dequantized:
    quantized = Float8PerHead().quantize(key_value)
    assert supported(query, quantized).eq(0).all()
    expected = torch_sdpa(query, quantized.dequantize(query.dtype))
    torch.testing.assert_close(torch_sdpa(query, quantized), expected)
