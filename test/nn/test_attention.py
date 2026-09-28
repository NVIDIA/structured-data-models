# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sdm.nn import (
    SDPA,
    Attention,
    QASSMax,
    SoftplusScale,
    TransformerBlock,
)
from sdm.testing import withCUDA


def reference_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
) -> Tensor:
    # Expand KV for MQA and GQA
    groups = query.size(-2) // key.size(-2)
    key = key.repeat_interleave(groups, dim=-2)
    value = value.repeat_interleave(groups, dim=-2)
    return F.scaled_dot_product_attention(
        query=query.transpose(-3, -2),
        key=key.transpose(-3, -2),
        value=value.transpose(-3, -2),
        attn_mask=attn_mask.unsqueeze(-3) if attn_mask is not None else None,
        scale=scale,
    ).transpose(-3, -2)


@withCUDA
def test_attention_transforms(device: torch.device) -> None:
    module = Attention(
        channels=4,
        num_query_heads=2,
        query_transform=torch.nn.Sequential(
            torch.nn.RMSNorm(2, eps=1e-6, device=device),
            SoftplusScale(2, device=device),
        ),
        key_transform=torch.nn.RMSNorm(2, eps=1e-6, device=device),
        scale=1.0,
        device=device,
    )
    query = torch.randn(2, 3, 4, device=device)

    assert module(query).shape == query.shape


@withCUDA
@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(None, id="mha"),
        pytest.param(1, id="mqa"),
        pytest.param(2, id="gqa"),
    ],
)
def test_sdpa(
    device: torch.device,
    num_key_value_heads: int | None,
) -> None:
    channels = 3
    num_query_heads = 4
    module = SDPA(
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_query_heads

    # Match torch SDPA for unbatched query, key, and value tensors.
    query = torch.randn(4, num_query_heads, channels, device=device)
    key = torch.randn(5, num_key_value_heads, channels, device=device)
    value = torch.randn(5, num_key_value_heads, channels, device=device)

    out = module(query=query, key=key, value=value)

    expected = reference_sdpa(query=query, key=key, value=value)
    torch.testing.assert_close(out, expected)

    # Broadcast batch dimensions and apply a boolean attention mask.
    query = torch.randn(2, 3, num_query_heads, channels, device=device)
    key = torch.randn(5, num_key_value_heads, channels, device=device)
    value = torch.randn(1, 5, num_key_value_heads, channels, device=device)
    attn_mask = torch.randint(0, 2, (3, 5), dtype=torch.bool, device=device)

    out = module(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    expected = reference_sdpa(
        query=query,
        key=key.expand(2, -1, -1, -1),
        value=value.expand(2, -1, -1, -1),
        attn_mask=attn_mask,
    )
    torch.testing.assert_close(out, expected)

    # Empty key/value sequences produce a zero attention result.
    query = torch.randn(2, 3, num_query_heads, channels, device=device)
    key = torch.empty(1, 0, num_key_value_heads, channels, device=device)
    value = torch.empty(2, 0, num_key_value_heads, channels, device=device)
    out = module(
        query=query,
        key=key,
        value=value,
    )
    assert out.size() == query.size()
    torch.testing.assert_close(out, torch.zeros_like(query))

    # Apply sequence lengths to mask keys.
    batch_size = 2
    query_len = 4
    key_value_len = 5
    query = torch.randn(
        batch_size,
        query_len,
        num_query_heads,
        channels,
        device=device,
    )
    key = torch.randn(
        batch_size,
        key_value_len,
        num_key_value_heads,
        channels,
        device=device,
    )
    value = torch.randn(
        batch_size,
        key_value_len,
        num_key_value_heads,
        channels,
        device=device,
    )
    value[0, 3:] = 1000
    value[1, 1:] = -1000
    seqused_key_value = torch.tensor([3, 1], dtype=torch.int32, device=device)

    out = module(
        query=query,
        key=key,
        value=value,
        seqused_key_value=seqused_key_value,
    )

    key_index = torch.arange(key_value_len, device=device).view(
        1, 1, key_value_len
    )
    attn_mask = key_index < seqused_key_value.view(batch_size, 1, 1)
    attn_mask = attn_mask.expand(batch_size, query_len, key_value_len)
    expected = reference_sdpa(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    torch.testing.assert_close(out, expected)

    # Test rows share the same in-context training examples.
    batch_size = 2
    num_test = 4
    num_queries = 1
    num_train = 5
    query = torch.randn(
        batch_size,
        num_test,
        num_queries,
        num_query_heads,
        channels,
        device=device,
    )
    key = torch.randn(
        batch_size,
        1,
        num_train,
        num_key_value_heads,
        channels,
        device=device,
    )
    value = torch.randn(
        batch_size,
        1,
        num_train,
        num_key_value_heads,
        channels,
        device=device,
    )

    out = module(query=query, key=key, value=value)
    expected = reference_sdpa(
        query=query,
        key=key.expand(-1, num_test, -1, -1, -1),
        value=value.expand(-1, num_test, -1, -1, -1),
    )
    torch.testing.assert_close(out, expected)


@withCUDA
def test_sdpa_scale(device: torch.device) -> None:
    channels = 4
    num_heads = 2
    scale = 1.0
    module = SDPA(num_query_heads=num_heads, scale=scale)
    query = torch.randn(2, 3, num_heads, channels, device=device)
    key = torch.randn(2, 5, num_heads, channels, device=device)
    value = torch.randn(2, 5, num_heads, channels, device=device)

    out = module(query=query, key=key, value=value)
    expected = reference_sdpa(
        query=query,
        key=key,
        value=value,
        scale=scale,
    )

    torch.testing.assert_close(out, expected)


def test_sdpa_errors() -> None:
    channels = 3
    num_heads = 2
    module = SDPA(num_query_heads=num_heads)

    query = torch.randn(1, 2, num_heads, channels)
    key = torch.randn(1, 2, num_heads, channels)
    value = torch.randn(1, 2, num_heads, channels)
    attn_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="Cannot pass both"):
        module(
            query=query,
            key=key,
            value=value,
            seqused_key_value=torch.tensor([1], dtype=torch.int32),
            attn_mask=attn_mask,
        )

    with pytest.raises(
        ValueError,
        match=r"`seqused_key_value` must have dtype torch\.int32",
    ):
        module(
            query=query,
            key=key,
            value=value,
            seqused_key_value=torch.tensor([1], dtype=torch.int64),
        )

    with pytest.raises(
        ValueError,
        match=r"`attn_mask` must have dtype torch\.bool",
    ):
        module(
            query=query,
            key=key,
            value=value,
            attn_mask=torch.ones(1, 2, 2, dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="must be divisible"):
        SDPA(num_query_heads=4, num_key_value_heads=3)


@withCUDA
@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(None, id="mha"),
        pytest.param(1, id="mqa"),
        pytest.param(2, id="gqa"),
    ],
)
@pytest.mark.parametrize("qassmax", [False, True])
def test_attention(
    device: torch.device,
    num_key_value_heads: int | None,
    qassmax: bool,
) -> None:
    channels = 8
    num_query_heads = 4
    module = Attention(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        query_scaling=QASSMax(
            channels // num_query_heads,
            num_query_heads,
            device=device,
        )
        if qassmax
        else None,
        device=device,
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_query_heads
    head_dim = channels // num_query_heads
    expected_qkv = (num_query_heads + 2 * num_key_value_heads) * head_dim
    assert module.qkv_lin.out_features == expected_qkv

    query = torch.randn(2, 4, channels, device=device)
    key_value = torch.randn(2, 5, channels, device=device)
    attn_mask = torch.randint(0, 2, (2, 4, 5), dtype=torch.bool, device=device)

    out = module(query=query, key_value=None)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    out = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    query = torch.randn(2, 4, channels, device=device)
    out = module(query=query, key_value=query)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


@pytest.mark.parametrize("qassmax", [False, True])
def test_attention_kv_cache(qassmax: bool) -> None:
    channels = 8
    num_heads = 2
    module = Attention(
        channels=channels,
        num_query_heads=num_heads,
        query_scaling=QASSMax(channels // num_heads, num_heads)
        if qassmax
        else None,
    )

    with torch.no_grad():
        module.out_lin.weight.copy_(torch.eye(channels))
        module.out_lin.bias.zero_()

    query = torch.randn(2, 3, channels)
    key_value = torch.randn(2, 5, channels)
    attn_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    ).expand(2, -1, -1)

    direct_out = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
    )
    cache_out, kv = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        return_key_value=True,
    )
    cached_out = module(
        query=query,
        key_value=kv,
        attn_mask=attn_mask,
    )

    self_out, self_kv = module(query=query, return_key_value=True)
    self_cached_out = module(query=query, key_value=self_kv)

    assert kv.key.size() == (2, 5, num_heads, channels // num_heads)
    assert kv.value.size() == (2, 5, num_heads, channels // num_heads)
    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(cached_out, direct_out)
    torch.testing.assert_close(self_cached_out, self_out)


def test_empty_query_chunking_preserves_unbroadcast_shape() -> None:
    channels = 8
    module = TransformerBlock(
        channels,
        num_query_heads=2,
        mlp=torch.nn.Identity(),
    ).eval()

    query = torch.randn(1, 0, channels)
    key_value = torch.randn(3, 5, channels)

    expected = module(query=query, key_value=key_value)
    with torch.no_grad():
        actual = module(
            query=query,
            key_value=key_value,
            batch_size_limit=1,
        )

    assert actual.size() == expected.size() == query.size()
    torch.testing.assert_close(actual, expected)


def test_attention_errors() -> None:
    channels = 6
    num_heads = 3
    module = Attention(channels=channels, num_query_heads=num_heads)
    query = torch.randn(2, 4, channels)

    with pytest.raises(ValueError, match="Cannot pass both"):
        module(
            query=query,
            seqused_key_value=torch.tensor([4, 4], dtype=torch.int32),
            attn_mask=torch.randint(0, 2, (2, 4, 5), dtype=torch.bool),
        )

    with pytest.raises(
        ValueError,
        match=r"`seqused_key_value` must have dtype torch\.int32",
    ):
        module(
            query=query,
            seqused_key_value=torch.tensor([4, 4], dtype=torch.int64),
        )

    with pytest.raises(
        ValueError,
        match=r"`attn_mask` must have dtype torch\.bool",
    ):
        module(
            query=query,
            attn_mask=torch.ones(2, 4, 4, dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="must be divisible"):
        Attention(channels=5, num_query_heads=2)


def test_attention_batch_size_limit_propagation() -> None:
    channels = 8
    num_heads = 2
    query = torch.randn(5, 3, channels)
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        mlp=torch.nn.Identity(),
        query_norm=torch.nn.LayerNorm(channels),
    ).eval()

    with torch.no_grad():
        module.attn.out_lin.weight.copy_(torch.eye(channels))
        module.attn.out_lin.bias.zero_()
    expected = module(query=query)

    outer_module = module.query_norm
    assert outer_module is not None
    outer_batch_sizes: list[int] = []
    handle = outer_module.register_forward_pre_hook(
        lambda _module, args, batch_sizes=outer_batch_sizes: (
            batch_sizes.append(args[0].size(0))
        )
    )
    with (
        patch.object(
            F,
            "scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa,
        torch.no_grad(),
    ):
        out = module(query=query, batch_size_limit=2)
    handle.remove()

    sdpa_batch_sizes = [
        call.kwargs["query"].size(0) for call in sdpa.call_args_list
    ]
    assert outer_batch_sizes == [2, 2, 1]
    assert sdpa_batch_sizes == [2, 2, 1]
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("requires_grad", "compiling"),
    [
        pytest.param(True, False, id="requires-grad"),
        pytest.param(False, True, id="compiling"),
    ],
)
def test_transformer_block_batch_size_limit_bypass(
    requires_grad: bool,
    compiling: bool,
) -> None:
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        mlp=torch.nn.Identity(),
        query_norm=torch.nn.LayerNorm(8),
    )
    query = torch.randn(5, 3, 8)
    batch_sizes: list[int] = []
    assert module.query_norm is not None
    handle = module.query_norm.register_forward_pre_hook(
        lambda _module, args: batch_sizes.append(args[0].size(0))
    )

    with (
        torch.set_grad_enabled(requires_grad),
        patch.object(
            torch.compiler,
            "is_compiling",
            return_value=compiling,
        ),
    ):
        module(query=query, batch_size_limit=2)
    handle.remove()

    assert batch_sizes == [5]


def test_return_key_value_positional_compatibility() -> None:
    channels = 8
    query = torch.randn(2, 3, channels)
    modules = (
        Attention(channels=channels, num_query_heads=2),
        TransformerBlock(
            channels=channels,
            num_query_heads=2,
            mlp=torch.nn.Identity(),
        ),
    )

    for module in modules:
        forward = cast(Callable[..., object], module.forward)
        result = forward(query, None, None, None, return_key_value=True)
        assert isinstance(result, tuple)
        assert len(result) == 2


@withCUDA
@pytest.mark.parametrize("qassmax", [False, True])
def test_transformer_block(device: torch.device, qassmax: bool) -> None:
    batch_size = 2
    query_len = 3
    key_value_len = 5
    channels = 8
    num_heads = 2
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        mlp=torch.nn.Sequential(
            torch.nn.Linear(channels, 16, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(16, channels, device=device),
        ),
        query_norm=torch.nn.LayerNorm(channels, device=device),
        key_value_norm=torch.nn.LayerNorm(channels, device=device),
        query_scaling=QASSMax(channels // num_heads, num_heads, device=device)
        if qassmax
        else None,
        device=device,
    )
    query = torch.randn(batch_size, query_len, channels, device=device)
    key_value = torch.randn(batch_size, key_value_len, channels, device=device)
    seqused_key_value = torch.tensor([3, 1], dtype=torch.int32, device=device)
    key_index = torch.arange(key_value_len, device=device).view(
        1, 1, key_value_len
    )
    attn_mask = key_index < seqused_key_value.view(batch_size, 1, 1)
    attn_mask = attn_mask.expand(batch_size, query_len, key_value_len)

    out = module(query=query)
    assert out.size() == query.size()
    assert out.dtype == query.dtype
    assert out.device == query.device

    out = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
    )
    assert out.size() == query.size()
    assert out.dtype == query.dtype
    assert out.device == query.device

    with torch.no_grad():
        module.attn.out_lin.weight.copy_(torch.eye(channels))
        module.attn.out_lin.bias.zero_()

    out1 = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
    )
    out2 = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
    )
    with torch.no_grad():
        chunked_out = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            batch_size_limit=1,
        )
        buffer = torch.empty_like(out1)
        buffered_out = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            out=buffer,
        )
        chunked_buffer = torch.empty_like(out1)
        chunked_buffered_out = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            batch_size_limit=1,
            out=chunked_buffer,
        )
    # Both paths reduce to the same boolean mask and SDPA kernel, but with
    # `qassmax` the key lengths enter :class:`QASSMax` as differently-shaped
    # tensors (`[..., 1]` from `seqused_key_value` vs `[..., Q]` from the
    # mask), so its MLP GEMMs may round differently in float32.
    if qassmax:
        torch.testing.assert_close(out1, out2, atol=5e-4, rtol=5e-3)
    else:
        torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(chunked_out, out1)
    assert buffered_out is buffer
    torch.testing.assert_close(buffered_out, out1)
    assert chunked_buffered_out is chunked_buffer
    torch.testing.assert_close(chunked_buffered_out, out1)

    # Test no padding leakage
    new_key_value = key_value.clone()
    new_key_value[0, 3:] = torch.randn_like(new_key_value[0, 3:]) * 1000
    new_key_value[1, 1:] = torch.randn_like(new_key_value[1, 1:]) * -1000
    out3 = module(
        query=query,
        key_value=new_key_value,
        seqused_key_value=seqused_key_value,
    )
    torch.testing.assert_close(out1, out3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transformer_block_chunked_noncontiguous_out(
    dtype: torch.dtype,
) -> None:
    channels = 8
    module = TransformerBlock(
        channels=channels,
        num_query_heads=2,
        mlp=torch.nn.Identity(),
    ).eval()

    base = torch.randn(2, 3, 4, channels)
    query = base.transpose(-2, -3)
    # Under autocast, outputs can differ in dtype from a preallocated buffer.
    buffer = torch.empty_like(base, dtype=dtype).transpose(-2, -3)

    assert not query.is_contiguous()
    assert not buffer.is_contiguous()

    expected = module(query=query)
    with torch.no_grad():
        actual = module(
            query=query,
            batch_size_limit=1,
            out=buffer,
        )

    assert actual is buffer
    torch.testing.assert_close(actual, expected.to(dtype))


def test_transformer_block_kv_cache() -> None:
    batch_size = 2
    query_len = 3
    key_value_len = 5
    channels = 8
    num_heads = 2
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        mlp=torch.nn.Identity(),
    )
    with torch.no_grad():
        module.attn.out_lin.weight.copy_(torch.eye(channels))
        module.attn.out_lin.bias.zero_()

    query = torch.randn(batch_size, query_len, channels)
    key_value = torch.randn(batch_size, key_value_len, channels)
    seqused_key_value = torch.tensor([3, 1], dtype=torch.int32)

    direct_out = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
    )
    cache_out, kv = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
        return_key_value=True,
    )
    cached_out = module(
        query=query,
        key_value=kv,
        seqused_key_value=seqused_key_value,
    )

    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(cached_out, direct_out)
