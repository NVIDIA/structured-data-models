# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn import SDPA, Float8PerHead, QuantizedTensor
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
    options = {
        "attn_mask": attn_mask,
        "seqused_key_value": seqused_key_value,
        "is_causal": False,
        "dropout_p": 0.0,
        "scale": 0.5,
    }
    assert backend.supports(query, key, value, **options)
    out = backend.forward(query, key, value, **options)
    torch.testing.assert_close(out, expected)

    fp8 = Float8PerHead()
    assert not backend.supports(query, fp8.quantize(key), value, **options)
    assert not backend.supports(query, key, fp8.quantize(value), **options)


class _ZeroBackend(AttentionBackend):
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def supports(
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        *,
        attn_mask: Tensor | None,
        seqused_key_value: Tensor | None,
        is_causal: bool,
        dropout_p: float,
        scale: float | None,
    ) -> bool:
        return (
            self.supported
            and isinstance(key, Tensor)
            and isinstance(value, Tensor)
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
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
    value = torch.randn_like(key)
    torch_sdpa = SDPA(num_query_heads=4)
    supported = SDPA(num_query_heads=4, backend=_ZeroBackend(True))
    unsupported = SDPA(num_query_heads=4, backend=_ZeroBackend(False))

    assert supported(query, key, value).eq(0).all()
    expected = torch_sdpa(query, key, value)
    torch.testing.assert_close(unsupported(query, key, value), expected)

    # Backends that can't read quantized keys or values get them dequantized:
    fp8 = Float8PerHead()
    quantized_key = fp8.quantize(key)
    quantized_value = fp8.quantize(value)
    dequantized_key = quantized_key.dequantize(query.dtype)
    dequantized_value = quantized_value.dequantize(query.dtype)
    assert supported(query, quantized_key, quantized_value).eq(0).all()
    out = torch_sdpa(query, quantized_key, quantized_value)
    expected = torch_sdpa(query, dequantized_key, dequantized_value)
    torch.testing.assert_close(out, expected)
    out = torch_sdpa(query, key, quantized_value)
    expected = torch_sdpa(query, key, dequantized_value)
    torch.testing.assert_close(out, expected)
