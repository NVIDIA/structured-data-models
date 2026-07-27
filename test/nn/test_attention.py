from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from sdm.cache import KVCacheEntry
from sdm.nn import (
    SDPA,
    Attention,
    QASSMax,
    RotaryEmbedding,
    TransformerBlock,
)
from sdm.testing import withCUDA
from torch import Tensor


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
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_qassmax(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    channels = 2
    num_heads = 3
    module = QASSMax(
        channels=channels,
        num_heads=num_heads,
        hidden_channels=4,
        device=device,
    )

    query = torch.randn(2, 3, num_heads, channels, device=device)

    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    out = module(query, key_len)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


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
        channels=channels,
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
    module = SDPA(
        channels=channels,
        num_query_heads=num_heads,
        scale=scale,
        device=device,
    )
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
    module = SDPA(channels=channels, num_query_heads=num_heads)

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
        SDPA(channels=4, num_query_heads=4, num_key_value_heads=3)

    with pytest.raises(ValueError, match="must be positive"):
        module(query=query, key=key, value=value, batch_size_limit=0)


def test_sdpa_batch_size_limit() -> None:
    channels = 3
    num_query_heads = 4
    num_key_value_heads = 2
    module = SDPA(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
    ).eval()

    query_len = 3
    key_value_len = 4
    query = torch.randn(2, 1, query_len, num_query_heads, channels)
    key = torch.randn(1, 3, key_value_len, num_key_value_heads, channels)
    value = torch.randn(2, 1, key_value_len, num_key_value_heads, channels)
    attn_mask = torch.randint(
        0,
        2,
        (1, 3, query_len, key_value_len),
        dtype=torch.bool,
    )
    attn_mask[..., 0] = True

    expected = module(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    with patch.object(
        F,
        "scaled_dot_product_attention",
        wraps=F.scaled_dot_product_attention,
    ) as sdpa:
        out = module(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            batch_size_limit=4,
        )

    batch_sizes = [
        (
            call.kwargs["query"].size(0),
            call.kwargs["attn_mask"].size(0),
        )
        for call in sdpa.call_args_list
    ]
    assert batch_sizes == [(4, 4), (2, 2)]
    assert out.size() == (2, 3, query_len, num_query_heads, channels)
    torch.testing.assert_close(out, expected)


def test_batch_size_limit_autocast_dtype() -> None:
    sdpa = SDPA(channels=3, num_query_heads=2).eval()
    query = torch.randn(5, 3, 2, 3)
    key = torch.randn(5, 4, 2, 3)
    value = torch.randn(5, 4, 2, 3)

    attention = Attention(channels=6, num_query_heads=2).eval()
    with torch.no_grad():
        attention.out_lin.weight.copy_(torch.eye(6))
        attention.out_lin.bias.zero_()
    attention_query = torch.randn(5, 3, 6)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected_sdpa = sdpa(query=query, key=key, value=value)
        chunked_sdpa = sdpa(
            query=query,
            key=key,
            value=value,
            batch_size_limit=2,
        )
        expected_attention = attention(query=attention_query)
        chunked_attention = attention(
            query=attention_query,
            batch_size_limit=2,
        )

    assert chunked_sdpa.dtype == expected_sdpa.dtype == torch.bfloat16
    assert (
        chunked_attention.dtype == expected_attention.dtype == torch.bfloat16
    )
    torch.testing.assert_close(chunked_sdpa, expected_sdpa)
    torch.testing.assert_close(chunked_attention, expected_attention)


@pytest.mark.parametrize(
    ("training", "compiling", "batch_size_limit"),
    [
        pytest.param(True, False, 2, id="training"),
        pytest.param(False, True, 2, id="compiling"),
        pytest.param(False, False, None, id="disabled"),
        pytest.param(False, False, 5, id="within-limit"),
    ],
)
def test_sdpa_batch_size_limit_bypass(
    training: bool,
    compiling: bool,
    batch_size_limit: int | None,
) -> None:
    module = SDPA(channels=3, num_query_heads=2)
    module.train(training)
    query = torch.randn(5, 3, 2, 3)
    key = torch.randn(5, 4, 2, 3)
    value = torch.randn(5, 4, 2, 3)

    with (
        patch.object(torch.compiler, "is_compiling", return_value=compiling),
        patch.object(
            F,
            "scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa,
    ):
        module(
            query=query,
            key=key,
            value=value,
            batch_size_limit=batch_size_limit,
        )

    assert sdpa.call_count == 1
    assert sdpa.call_args.kwargs["query"].size(0) == 5


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
@pytest.mark.parametrize("rope", [False, True])
def test_attention(
    device: torch.device,
    num_key_value_heads: int | None,
    qassmax: bool,
    rope: bool,
) -> None:
    channels = 8
    num_query_heads = 4
    dtype = torch.float32
    module = Attention(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        qassmax=qassmax,
        device=device,
        dtype=dtype,
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_query_heads
    head_dim = channels // num_query_heads
    expected_qkv = (num_query_heads + 2 * num_key_value_heads) * head_dim
    assert module.qkv_lin.out_features == expected_qkv

    query = torch.randn(2, 4, channels, dtype=dtype, device=device)
    key_value = torch.randn(2, 5, channels, dtype=dtype, device=device)
    attn_mask = torch.randint(0, 2, (2, 4, 5), dtype=torch.bool, device=device)

    rotary_embedding: RotaryEmbedding | None = None
    if rope:
        rotary_embedding = RotaryEmbedding(
            channels=head_dim,
            layout="split_half",
            device=device,
            dtype=dtype,
        )

    out = module(query=query, key_value=None, rope=rotary_embedding)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    out = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    query = torch.randn(2, 4, channels, dtype=dtype, device=device)
    out = module(query=query, key_value=query, rope=rotary_embedding)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


@pytest.mark.parametrize("qassmax", [False, True])
@pytest.mark.parametrize("rope", [False, True])
def test_attention_kv_cache(qassmax: bool, rope: bool) -> None:
    channels = 8
    num_heads = 2
    module = Attention(
        channels=channels,
        num_query_heads=num_heads,
        qassmax=qassmax,
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

    rotary_embedding: RotaryEmbedding | None = None
    if rope:
        rotary_embedding = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
        )

    direct_out = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
    )
    cache_out, kv = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
        return_key_value=True,
    )
    cached_out = module(
        query=query,
        key_value=kv,
        attn_mask=attn_mask,
        rope=rotary_embedding,
    )
    module.eval()
    chunked_cache_out, chunked_kv = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
        return_key_value=True,
        batch_size_limit=1,
    )
    chunked_cached_out = module(
        query=query,
        key_value=kv,
        attn_mask=attn_mask,
        rope=rotary_embedding,
        batch_size_limit=1,
    )

    self_out, self_kv = module(query=query, return_key_value=True)
    self_cached_out = module(query=query, key_value=self_kv)

    assert kv.key.size() == (2, 5, num_heads, channels // num_heads)
    assert kv.value.size() == (2, 5, num_heads, channels // num_heads)
    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(chunked_cache_out, direct_out)
    torch.testing.assert_close(chunked_kv.key, kv.key)
    torch.testing.assert_close(chunked_kv.value, kv.value)
    torch.testing.assert_close(cached_out, direct_out)
    torch.testing.assert_close(chunked_cached_out, direct_out)
    torch.testing.assert_close(self_cached_out, self_out)


def test_attention_empty_query_chunked_kv_cache() -> None:
    batch_size = 3
    channels = 8
    num_heads = 2
    module = Attention(channels=channels, num_query_heads=num_heads).eval()
    query = torch.randn(batch_size, 0, channels)
    key_value = torch.randn(batch_size, 5, channels)

    expected_out, expected_kv = module(
        query=query,
        key_value=key_value,
        return_key_value=True,
    )
    out, kv = module(
        query=query,
        key_value=key_value,
        return_key_value=True,
        batch_size_limit=2,
    )

    assert out.size() == expected_out.size() == query.size()
    torch.testing.assert_close(kv.key, expected_kv.key)
    torch.testing.assert_close(kv.value, expected_kv.value)


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: Attention(channels=8, num_query_heads=2),
        lambda: TransformerBlock(
            channels=8,
            num_query_heads=2,
            feedforward_channels=16,
        ),
    ],
    ids=["attention", "transformer-block"],
)
def test_empty_query_chunking_preserves_unbroadcast_shape(
    module_factory: Callable[[], Attention | TransformerBlock],
) -> None:
    channels = 8
    module = module_factory().eval()
    query = torch.randn(1, 0, channels)
    key_value = torch.randn(3, 5, channels)

    expected = module(query=query, key_value=key_value)
    actual = module(
        query=query,
        key_value=key_value,
        batch_size_limit=1,
    )

    assert actual.size() == expected.size() == query.size()
    torch.testing.assert_close(actual, expected)


def test_attention_chunked_broadcast_kv_cache_shape() -> None:
    channels = 8
    num_heads = 2
    module = Attention(channels=channels, num_query_heads=num_heads).eval()
    query = torch.randn(2, 3, 4, channels)
    key_value = torch.randn(2, 1, 5, channels)

    expected_out, expected_kv = module(
        query=query,
        key_value=key_value,
        return_key_value=True,
    )
    out, kv = module(
        query=query,
        key_value=key_value,
        return_key_value=True,
        batch_size_limit=2,
    )

    assert out.size() == expected_out.size() == query.size()
    assert kv.key.size() == (2, 1, 5, num_heads, channels // num_heads)
    assert kv.value.size() == kv.key.size()
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(kv.key, expected_kv.key)
    torch.testing.assert_close(kv.value, expected_kv.value)


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
    modules = (
        Attention(channels=channels, num_query_heads=num_heads),
        TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            feedforward_channels=16,
        ),
    )

    for module in modules:
        module.eval()
        attention = module if isinstance(module, Attention) else module.attn
        with torch.no_grad():
            attention.out_lin.weight.copy_(torch.eye(channels))
            attention.out_lin.bias.zero_()
        expected = module(query=query)

        outer_module = (
            module.qkv_lin if isinstance(module, Attention) else module.q_norm
        )
        outer_batch_sizes: list[int] = []
        handle = outer_module.register_forward_pre_hook(
            lambda _module, args, batch_sizes=outer_batch_sizes: (
                batch_sizes.append(args[0].size(0))
            )
        )
        with patch.object(
            F,
            "scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            out = module(query=query, batch_size_limit=2)
        handle.remove()

        sdpa_batch_sizes = [
            call.kwargs["query"].size(0) for call in sdpa.call_args_list
        ]
        assert outer_batch_sizes == [2, 2, 1]
        assert sdpa_batch_sizes == [2, 2, 1]
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("training", "compiling"),
    [
        pytest.param(True, False, id="training"),
        pytest.param(False, True, id="compiling"),
    ],
)
def test_transformer_block_batch_size_limit_bypass(
    training: bool,
    compiling: bool,
) -> None:
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
    )
    module.train(training)
    query = torch.randn(5, 3, 8)
    batch_sizes: list[int] = []
    handle = module.q_norm.register_forward_pre_hook(
        lambda _module, args: batch_sizes.append(args[0].size(0))
    )

    with patch.object(
        torch.compiler,
        "is_compiling",
        return_value=compiling,
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
            feedforward_channels=16,
        ),
    )

    for module in modules:
        forward = cast(Callable[..., object], module.forward)
        result = forward(query, None, None, None, None, True)
        assert isinstance(result, tuple)
        assert len(result) == 2


@withCUDA
@pytest.mark.parametrize("norm", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("qassmax", [False, True])
@pytest.mark.parametrize("rope", [False, True])
def test_transformer_block(
    device: torch.device,
    norm: str,
    qassmax: bool,
    rope: bool,
) -> None:
    batch_size = 2
    query_len = 3
    key_value_len = 5
    channels = 8
    num_heads = 2
    feedforward_channels = 16
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        feedforward_channels=feedforward_channels,
        qassmax=qassmax,
        norm=norm,
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

    rotary_embedding: RotaryEmbedding | None = None
    if rope:
        rotary_embedding = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            device=device,
        )

    out = module(query=query, rope=rotary_embedding)
    assert out.size() == query.size()
    assert out.dtype == query.dtype
    assert out.device == query.device

    out = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
        rope=rotary_embedding,
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
        rope=rotary_embedding,
    )
    out2 = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
    )
    module.eval()
    chunked_out = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
        rope=rotary_embedding,
        batch_size_limit=1,
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

    # Test no padding leakage
    new_key_value = key_value.clone()
    new_key_value[0, 3:] = torch.randn_like(new_key_value[0, 3:]) * 1000
    new_key_value[1, 1:] = torch.randn_like(new_key_value[1, 1:]) * -1000
    out3 = module(
        query=query,
        key_value=new_key_value,
        seqused_key_value=seqused_key_value,
        rope=rotary_embedding,
    )
    torch.testing.assert_close(out1, out3)


def test_transformer_block_norm_kwargs_precedence() -> None:
    # User-provided `norm_kwargs` win over the `device`/`dtype` arguments.
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
        norm_kwargs={"dtype": torch.float64},
    )
    for norm in (module.q_norm, module.kv_norm, module.mlp[0]):
        assert next(norm.parameters()).dtype == torch.float64

    # The `dtype` argument still applies when `norm_kwargs` does not set it.
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
        norm_kwargs={"eps": 1e-6},
        dtype=torch.float64,
    )
    for norm in (module.q_norm, module.kv_norm, module.mlp[0]):
        assert next(norm.parameters()).dtype == torch.float64


def test_transformer_block_norm_callable() -> None:
    # A norm class is instantiated per site, so no site shares an instance.
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
        norm=torch.nn.RMSNorm,
        norm_kwargs={"eps": 1e-6, "dtype": torch.float64},
    )
    norms = (module.q_norm, module.kv_norm, module.mlp[0])
    for norm in norms:
        assert isinstance(norm, torch.nn.RMSNorm)
        assert norm.normalized_shape == (8,)
        assert norm.eps == 1e-6
        assert next(norm.parameters()).dtype == torch.float64
    assert len({id(norm) for norm in norms}) == 3

    param_ids = [{id(param) for param in norm.parameters()} for norm in norms]
    assert param_ids[0].isdisjoint(param_ids[1])
    assert param_ids[0].isdisjoint(param_ids[2])
    assert param_ids[1].isdisjoint(param_ids[2])


def test_transformer_block_kv_cache() -> None:
    batch_size = 2
    query_len = 3
    key_value_len = 5
    channels = 8
    num_heads = 2
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        feedforward_channels=16,
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


def test_attention_key_value_cache_dtype_mismatch() -> None:
    module = Attention(channels=8, num_query_heads=2)
    query = torch.randn(2, 3, 8)
    cached = KVCacheEntry(
        key=torch.randn(2, 5, 2, 4, dtype=torch.bfloat16),
        value=torch.randn(2, 5, 2, 4, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="Re-run the caching step"):
        module(query=query, key_value=cached)


@withCUDA
def test_attention_key_value_cache_autocast(device: torch.device) -> None:
    module = Attention(channels=8, num_query_heads=2, device=device)
    query = torch.randn(2, 3, 8, device=device)
    key_value = torch.randn(2, 5, 8, device=device)

    # Caching and replaying under the same autocast context is valid even
    # though the pre-projection query stays float32: the dtype guard compares
    # against the projected query, so autocast validates. Runs on cpu too so
    # this guard decision keeps CPU-CI coverage.
    with torch.amp.autocast(device.type, torch.bfloat16):
        out, cached = module(
            query=query, key_value=key_value, return_key_value=True
        )
        replayed = module(query=query, key_value=cached)
    torch.testing.assert_close(replayed, out)
