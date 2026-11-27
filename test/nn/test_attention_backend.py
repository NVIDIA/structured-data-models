# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.cache import Float8PerHead, QuantizedTensor
from sdm.nn import SDPA
from sdm.nn.attention_backend import FP8AttentionBackend, TorchAttentionBackend
from sdm.testing import onlyCUDA


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


class MyAttentionBackend(TorchAttentionBackend):
    def forward(
        self,
        query: Tensor,
        key: Tensor | QuantizedTensor,
        value: Tensor | QuantizedTensor,
        **kwargs: object,
    ) -> Tensor:
        return torch.zeros_like(query, dtype=cast(Tensor, value).dtype)


def test_sdpa_backend() -> None:
    query = torch.randn(2, 3, 5, 4, 8)  # [..., Q, Hq, C]
    key = torch.randn(2, 3, 7, 4, 8)  # [..., KV, Hkv, C]
    value = torch.randn_like(key)
    sdpa = SDPA(num_query_heads=4, backend="torch")
    sdpa.backend = MyAttentionBackend()
    fa3 = SDPA(num_query_heads=4, backend="fa3")

    assert sdpa(query, key, value).eq(0).all()
    with pytest.raises(RuntimeError, match="FlashAttention3Backend does not"):
        fa3(query, key, value, is_causal=True)

    # Backends that can't read quantized keys or values get them dequantized:
    torch_sdpa = SDPA(num_query_heads=4)
    fp8 = Float8PerHead()
    quantized_key = fp8.quantize(key)
    quantized_value = fp8.quantize(value)
    dequantized_key = quantized_key.dequantize(query.dtype)
    dequantized_value = quantized_value.dequantize(query.dtype)
    assert sdpa(query, quantized_key, quantized_value).eq(0).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = sdpa(query, quantized_key, quantized_value)
    assert out.dtype == torch.bfloat16
    out = torch_sdpa(query, quantized_key, quantized_value)
    expected = torch_sdpa(query, dequantized_key, dequantized_value)
    torch.testing.assert_close(out, expected)
    out = torch_sdpa(query, key, quantized_value)
    expected = torch_sdpa(query, key, dequantized_value)
    torch.testing.assert_close(out, expected)


@onlyCUDA
@pytest.mark.parametrize(
    ("channels", "context"),
    [(32, 513), (64, 513), (128, 513), (128, 32768), (256, 513)],
)
def test_fp8_attention_backend(channels: int, context: int) -> None:
    pytest.importorskip("triton")
    capability = torch.cuda.get_device_capability()
    if capability not in {(8, 9), (9, 0), (12, 0), (12, 1)}:
        pytest.skip("FP8AttentionBackend does not support this GPU")
    query = torch.randn(
        2, 129, 8, channels, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        2, context, 2, channels, device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    # Give each batch element and head its own magnitude, so that a mixed-up
    # scale changes the output:
    query *= torch.linspace(0.25, 1, 16, device="cuda").view(2, 1, 8, 1)
    key *= key.new_tensor([0.5, 1, 0.25, 0.75]).view(2, 1, 2, 1)
    value *= value.new_tensor([0.01, 1, 100, 0.1]).view(2, 1, 2, 1)
    fp8 = Float8PerHead()
    quantized_key, quantized_value = fp8.quantize(key), fp8.quantize(value)
    options = {
        "attn_mask": None,
        "seqused_key_value": None,
        "is_causal": False,
        "dropout_p": 0.0,
        "scale": None,
    }
    backend = FP8AttentionBackend()
    assert not backend.supports(query, key, value, **options)
    assert not backend.supports(query, quantized_key, value, **options)
    assert not backend.supports(
        query=query,
        key=QuantizedTensor.quantize(key, dim=(0, 1, 2, 3)),
        value=quantized_value,
        **options,
    )
    requires_grad = query.detach().requires_grad_()
    assert not backend.supports(
        query=requires_grad,
        key=quantized_key,
        value=quantized_value,
        **options,
    )
    assert not backend.supports(
        query=query,
        key=fp8.quantize(key.detach().requires_grad_()),
        value=quantized_value,
        **options,
    )
    supports = torch.compile(backend.supports, fullgraph=True)
    assert not supports(
        query=query,
        key=quantized_key,
        value=quantized_value,
        **options,
    )
    assert backend.supports(
        query=query,
        key=quantized_key,
        value=quantized_value,
        **options,
    )

    with torch.inference_mode():
        out = backend.forward(
            query=query,
            key=quantized_key,
            value=quantized_value,
            **options,
        )
        expected = TorchAttentionBackend().forward(
            query=query,
            key=quantized_key.dequantize(query.dtype),
            value=quantized_value.dequantize(query.dtype),
            **options,
        )
    # Relative errors per batch element and head:
    error = (out.float() - expected.float()).square().mean(dim=(1, 3)).sqrt()
    norm = expected.float().square().mean(dim=(1, 3)).sqrt()
    assert error.div(norm).max() < 0.08
