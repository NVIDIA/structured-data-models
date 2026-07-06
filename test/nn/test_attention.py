from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
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
    ("num_key_value_heads", "enable_gqa"),
    [
        pytest.param(None, False, id="mha"),
        pytest.param(1, True, id="mqa"),
        pytest.param(2, True, id="gqa"),
    ],
)
def test_sdpa(
    device: torch.device,
    num_key_value_heads: int | None,
    enable_gqa: bool,
) -> None:
    channels = 3
    num_query_heads = 4
    module = SDPA(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
    )
    key_value_heads = (
        num_query_heads if num_key_value_heads is None else num_key_value_heads
    )

    # Match torch SDPA for unbatched query, key, and value tensors.
    query = torch.randn(4, num_query_heads, channels, device=device)
    key = torch.randn(5, key_value_heads, channels, device=device)
    value = torch.randn(5, key_value_heads, channels, device=device)

    with patch(
        "sdm.nn.attention.F.scaled_dot_product_attention",
        wraps=F.scaled_dot_product_attention,
    ) as mock_sdpa:
        out = module(query=query, key=key, value=value)
    mock_sdpa.assert_called_once()
    assert mock_sdpa.call_args.kwargs["enable_gqa"] is enable_gqa

    expected = reference_sdpa(query=query, key=key, value=value)
    torch.testing.assert_close(out, expected)

    # Broadcast batch dimensions and apply a boolean attention mask.
    query = torch.randn(2, 3, num_query_heads, channels, device=device)
    key = torch.randn(5, key_value_heads, channels, device=device)
    value = torch.randn(1, 5, key_value_heads, channels, device=device)
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
        key_value_heads,
        channels,
        device=device,
    )
    value = torch.randn(
        batch_size,
        key_value_len,
        key_value_heads,
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
        key_value_heads,
        channels,
        device=device,
    )
    value = torch.randn(
        batch_size,
        1,
        num_train,
        key_value_heads,
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
    key_value_heads = (
        num_query_heads if num_key_value_heads is None else num_key_value_heads
    )
    head_dim = channels // num_query_heads
    expected_qkv = (num_query_heads + 2 * key_value_heads) * head_dim
    assert module.qkv_lin.out_features == expected_qkv

    query = torch.randn(2, 4, channels, dtype=dtype, device=device)
    key_value = torch.randn(2, 5, channels, dtype=dtype, device=device)
    attn_mask = torch.randint(0, 2, (2, 4, 5), dtype=torch.bool, device=device)

    rotary_embedding: RotaryEmbedding | None = None
    if rope:
        rotary_embedding = RotaryEmbedding(
            channels=head_dim,
            device=device,
            dtype=dtype,
        )

    out = module(query=query, key_value=None, rope=rotary_embedding)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    out, kv = module(
        query=query,
        key_value=key_value,
        attn_mask=attn_mask,
        rope=rotary_embedding,
        return_key_value=True,
    )
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device
    assert kv.key.size() == (2, 5, key_value_heads, head_dim)
    assert kv.value.size() == (2, 5, key_value_heads, head_dim)

    cached_out = module(
        query=query,
        key_value=kv,
        attn_mask=attn_mask,
        rope=rotary_embedding,
    )
    assert cached_out.shape == query.shape

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
        rotary_embedding = RotaryEmbedding(channels=channels // num_heads)

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

    self_out, self_kv = module(query=query, return_key_value=True)
    self_cached_out = module(query=query, key_value=self_kv)

    assert kv.key.size() == (2, 5, num_heads, channels // num_heads)
    assert kv.value.size() == (2, 5, num_heads, channels // num_heads)
    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(cached_out, direct_out)
    torch.testing.assert_close(self_cached_out, self_out)


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


@withCUDA
@pytest.mark.parametrize("num_key_value_heads", [1, 2, 4])
@pytest.mark.parametrize("rope_on", [False, True])
def test_attention_gqa_numerical(
    device: torch.device,
    num_key_value_heads: int,
    rope_on: bool,
) -> None:
    # float64 makes the comparison exact and TF32-immune: the module computes
    # K/V via one fused F.linear while the reference below uses separate
    # projections, and on some GPUs differently-shaped fp32 (TF32) matmuls
    # round differently. fp64 still catches any q/k/v mis-slice or swap.
    dtype = torch.float64
    channels = 8
    num_query_heads = 4
    head_dim = channels // num_query_heads
    q_dim = num_query_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    module = Attention(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        device=device,
        dtype=dtype,
    )
    # Randomize projections (out_lin is zero-initialized) so distinct query,
    # key, and value weights make any q/k/v mis-slice or swap fail allclose.
    with torch.no_grad():
        module.qkv_lin.weight.normal_()
        module.qkv_lin.bias.normal_()
        module.out_lin.weight.normal_()
        module.out_lin.bias.normal_()

    query = torch.randn(2, 3, channels, device=device, dtype=dtype)
    key_value = torch.randn(2, 5, channels, device=device, dtype=dtype)
    rope = (
        RotaryEmbedding(channels=head_dim, device=device, dtype=dtype)
        if rope_on
        else None
    )

    out = module(query=query, key_value=key_value, rope=rope)

    # Hand-built reference using the module's own projection weights.
    qw, kw, vw = module.qkv_lin.weight.split([q_dim, kv_dim, kv_dim], dim=0)
    qb, kb, vb = module.qkv_lin.bias.split([q_dim, kv_dim, kv_dim], dim=0)
    q = F.linear(query, qw, qb).unflatten(-1, (num_query_heads, head_dim))
    k = F.linear(key_value, kw, kb).unflatten(
        -1, (num_key_value_heads, head_dim)
    )
    v = F.linear(key_value, vw, vb).unflatten(
        -1, (num_key_value_heads, head_dim)
    )
    if rope is not None:
        q = rope(q)
        k = rope(k)

    ref = reference_sdpa(query=q, key=k, value=v)
    ref = module.out_lin(ref.flatten(-2, -1))
    torch.testing.assert_close(out, ref)


@withCUDA
@pytest.mark.parametrize("qassmax", [False, True])
@pytest.mark.parametrize("rope", [False, True])
def test_transformer_block(
    device: torch.device,
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
    # Both paths reduce to the same boolean mask and SDPA kernel, but with
    # `qassmax` the key lengths enter :class:`QASSMax` as differently-shaped
    # tensors (`[..., 1]` from `seqused_key_value` vs `[..., Q]` from the
    # mask), so its MLP GEMMs may round differently in float32. A real
    # regression would be O(1).
    tolerance = {"atol": 5e-4, "rtol": 5e-3} if qassmax else {}
    torch.testing.assert_close(out1, out2, **tolerance)

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
