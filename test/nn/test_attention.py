from collections.abc import Callable
from typing import cast
from unittest import mock
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


def _noncontiguous_randn(
    *size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a strided tensor with the requested logical shape."""
    return torch.randn(*size, 2, device=device, dtype=dtype)[..., 0]


def _randomize_attention_exit(module: Attention) -> None:
    # Attention output projections are intentionally zero-initialized. Tests
    # comparing chunked execution need a non-trivial output to exercise the
    # projected attention result rather than comparing two tensors of zeros.
    with torch.no_grad():
        module.out_lin.weight.normal_(std=0.1)
        module.out_lin.bias.normal_(std=0.1)


def _randomize_block_exits(module: TransformerBlock) -> None:
    _randomize_attention_exit(module.attn)
    mlp_exit = module.mlp[-1]
    assert isinstance(mlp_exit, torch.nn.Linear)
    with torch.no_grad():
        mlp_exit.weight.normal_(std=0.1)
        mlp_exit.bias.normal_(std=0.1)


def _assert_sdpa_calls_bounded(
    spy: mock.Mock,
    limit: int,
    *,
    exact_calls: int | None = None,
) -> None:
    """Assert each SDPA call's broadcast batch fits the requested limit."""
    assert spy.call_count > 1
    if exact_calls is not None:
        assert spy.call_count == exact_calls

    for call in spy.call_args_list:
        query = call.kwargs.get("query", call.args[0] if call.args else None)
        key = call.kwargs.get(
            "key", call.args[1] if len(call.args) > 1 else None
        )
        value = call.kwargs.get(
            "value", call.args[2] if len(call.args) > 2 else None
        )
        assert isinstance(query, Tensor)
        assert isinstance(key, Tensor)
        assert isinstance(value, Tensor)
        batch_shape = torch.broadcast_shapes(
            query.shape[:-3], key.shape[:-3], value.shape[:-3]
        )
        assert batch_shape.numel() <= limit


_CACHE_BATCH_LAYOUTS = [
    pytest.param((5,), (5,), id="aligned"),
    pytest.param((5,), (1,), id="singleton"),
    pytest.param((2, 1), (1, 3), id="cross-broadcast"),
]


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
@pytest.mark.parametrize("auxiliary", ["mask", "seqused"])
def test_sdpa_batch_size_limit_broadcast(
    device: torch.device,
    auxiliary: str,
) -> None:
    """Direct SDPA chunks the fully broadcast batch without changing math."""
    dtype = torch.float64
    num_query_heads = 4
    num_key_value_heads = 2
    head_channels = 2
    query_len = 3
    key_value_len = 5
    limit = 2
    module = SDPA(
        channels=head_channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        qassmax=True,
        device=device,
        dtype=dtype,
    )

    # [A, 1] x [1, B] exercises arbitrary cross-broadcasting. Strided inputs
    # ensure the implementation cannot rely on a contiguous flattening view.
    query = _noncontiguous_randn(
        2,
        1,
        query_len,
        num_query_heads,
        head_channels,
        device=device,
        dtype=dtype,
    )
    key = _noncontiguous_randn(
        1,
        3,
        key_value_len,
        num_key_value_heads,
        head_channels,
        device=device,
        dtype=dtype,
    )
    value = _noncontiguous_randn(
        1,
        3,
        key_value_len,
        num_key_value_heads,
        head_channels,
        device=device,
        dtype=dtype,
    )

    attn_mask: Tensor | None = None
    seqused_key_value: Tensor | None = None
    if auxiliary == "mask":
        # The mask follows the query batch and broadcasts over the K/V batch.
        attn_mask = torch.randint(
            0,
            2,
            (2, 1, query_len, key_value_len),
            dtype=torch.bool,
            device=device,
        )
        attn_mask[..., 0] = True
    else:
        # Valid lengths follow the K/V batch and broadcast over the query.
        seqused_key_value = torch.tensor(
            [[key_value_len, 3, 1]], dtype=torch.int32, device=device
        )

    with torch.no_grad():
        expected = module(
            query=query,
            key=key,
            value=value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
        )
        with mock.patch.object(
            F,
            "scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as spy:
            out = module(
                query=query,
                key=key,
                value=value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                batch_size_limit=limit,
            )

    assert out.shape == (2, 3, query_len, num_query_heads, head_channels)
    _assert_sdpa_calls_bounded(spy, limit)
    torch.testing.assert_close(out, expected)

    if auxiliary == "mask":
        # The limit is inference-only and eager-only.
        with mock.patch.object(
            F,
            "scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as spy:
            grad_enabled = module(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                batch_size_limit=limit,
            )
        assert spy.call_count == 1
        torch.testing.assert_close(grad_enabled, expected)

        with (
            torch.no_grad(),
            mock.patch.object(
                torch.compiler, "is_compiling", return_value=True
            ),
            mock.patch.object(
                F,
                "scaled_dot_product_attention",
                wraps=F.scaled_dot_product_attention,
            ) as spy,
        ):
            compiling = module(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                batch_size_limit=limit,
            )
        assert spy.call_count == 1
        torch.testing.assert_close(compiling, expected)


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
    with pytest.raises(ValueError, match="must be positive"):
        module(query=query, key=key, value=value, batch_size_limit=-1)


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


@withCUDA
@pytest.mark.parametrize(
    ("query_batch", "key_value_batch"),
    _CACHE_BATCH_LAYOUTS,
)
@pytest.mark.parametrize("auxiliary", ["mask", "seqused"])
def test_attention_batch_size_limit_kv_cache(
    device: torch.device,
    query_batch: tuple[int, ...],
    key_value_batch: tuple[int, ...],
    auxiliary: str,
) -> None:
    """Cache recording/replay preserve native cache and broadcast semantics."""
    dtype = torch.float64
    channels = 8
    num_query_heads = 4
    num_key_value_heads = 2
    query_len = 3
    key_value_len = 5
    limit = 2
    module = Attention(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        qassmax=True,
        device=device,
        dtype=dtype,
    )
    _randomize_attention_exit(module)
    rope = RotaryEmbedding(
        channels=channels // num_query_heads,
        layout="split_half",
        device=device,
        dtype=dtype,
    )

    query = _noncontiguous_randn(
        *query_batch,
        query_len,
        channels,
        device=device,
        dtype=dtype,
    )
    key_value = _noncontiguous_randn(
        *key_value_batch,
        key_value_len,
        channels,
        device=device,
        dtype=dtype,
    )

    attn_mask: Tensor | None = None
    seqused_key_value: Tensor | None = None
    if auxiliary == "mask":
        attn_mask = torch.randint(
            0,
            2,
            (*query_batch, query_len, key_value_len),
            dtype=torch.bool,
            device=device,
        )
        attn_mask[..., 0] = True
    else:
        seqused_key_value = torch.randint(
            1,
            key_value_len + 1,
            key_value_batch,
            dtype=torch.int32,
            device=device,
        )

    output_batch = torch.broadcast_shapes(query_batch, key_value_batch)
    exact_calls = None
    if len(query_batch) == 1 and query_batch == key_value_batch:
        exact_calls = (output_batch.numel() + limit - 1) // limit
    cache_tail = (
        key_value_len,
        num_key_value_heads,
        channels // num_query_heads,
    )

    with torch.no_grad():
        expected, expected_kv = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
            rope=rope,
            return_key_value=True,
        )

        with mock.patch.object(
            module.sdpa, "forward", wraps=module.sdpa.forward
        ) as spy:
            recorded, kv = module(
                query=query,
                key_value=key_value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                rope=rope,
                batch_size_limit=limit,
                return_key_value=True,
            )
        _assert_sdpa_calls_bounded(spy, limit, exact_calls=exact_calls)

        # Recording must retain the native K/V batch shape. In particular, a
        # singleton/shared cache must not be expanded to the output batch.
        assert kv.key.shape == (*key_value_batch, *cache_tail)
        assert kv.value.shape == (*key_value_batch, *cache_tail)
        torch.testing.assert_close(kv.key, expected_kv.key)
        torch.testing.assert_close(kv.value, expected_kv.value)
        torch.testing.assert_close(recorded, expected)

        key_before = kv.key.clone()
        value_before = kv.value.clone()
        key_ptr = kv.key.data_ptr()
        value_ptr = kv.value.data_ptr()
        with mock.patch.object(
            module.sdpa, "forward", wraps=module.sdpa.forward
        ) as spy:
            replayed = module(
                query=query,
                key_value=kv,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                rope=rope,
                batch_size_limit=limit,
            )
        _assert_sdpa_calls_bounded(spy, limit, exact_calls=exact_calls)

    assert replayed.shape == (*output_batch, query_len, channels)
    torch.testing.assert_close(replayed, expected)
    # Replay treats the cache as immutable and keeps its compact allocation.
    assert kv.key.data_ptr() == key_ptr
    assert kv.value.data_ptr() == value_ptr
    torch.testing.assert_close(kv.key, key_before)
    torch.testing.assert_close(kv.value, value_before)
    assert recorded.abs().max() > 0


@withCUDA
def test_attention_kv_cache_batch_size_limit_autocast_dtype(
    device: torch.device,
) -> None:
    module = Attention(
        channels=8,
        num_query_heads=2,
        device=device,
        dtype=torch.float32,
    )
    _randomize_attention_exit(module)
    query = torch.randn(5, 3, 8, device=device)
    key_value = torch.randn(5, 4, 8, device=device)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=autocast_dtype),
    ):
        expected, expected_kv = module(
            query=query,
            key_value=key_value,
            return_key_value=True,
        )
        recorded, kv = module(
            query=query,
            key_value=key_value,
            return_key_value=True,
            batch_size_limit=2,
        )
        replayed = module(
            query=query,
            key_value=kv,
            batch_size_limit=2,
        )

    assert recorded.dtype == replayed.dtype == expected.dtype
    assert recorded.dtype == autocast_dtype
    assert kv.key.dtype == expected_kv.key.dtype == autocast_dtype
    assert kv.value.dtype == expected_kv.value.dtype == autocast_dtype
    torch.testing.assert_close(recorded, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(replayed, expected, atol=2e-2, rtol=2e-2)


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

    import contextlib

    # The grad gate would bypass chunking on its own; disable grads in the
    # compiling case so the is_compiling clause is what is exercised.
    grad_ctx = torch.no_grad() if compiling else contextlib.nullcontext()
    with (
        grad_ctx,
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
            feedforward_channels=16,
        ),
    )

    for module in modules:
        forward = cast(Callable[..., object], module.forward)
        result = forward(query, None, None, None, None, True)
        assert isinstance(result, tuple)
        assert len(result) == 2


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


@withCUDA
@pytest.mark.parametrize(
    ("query_batch", "key_value_batch"),
    _CACHE_BATCH_LAYOUTS,
)
def test_transformer_block_batch_size_limit_kv_cache(
    device: torch.device,
    query_batch: tuple[int, ...],
    key_value_batch: tuple[int, ...],
) -> None:
    """The entire block chunks while recording and replaying compact K/V."""
    dtype = torch.float64
    channels = 8
    num_query_heads = 4
    num_key_value_heads = 2
    query_len = 3
    key_value_len = 5
    limit = 2
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        feedforward_channels=16,
        qassmax=True,
        device=device,
        dtype=dtype,
    )
    _randomize_block_exits(module)
    rope = RotaryEmbedding(
        channels=channels // num_query_heads,
        layout="split_half",
        device=device,
        dtype=dtype,
    )

    query = _noncontiguous_randn(
        *query_batch,
        query_len,
        channels,
        device=device,
        dtype=dtype,
    )
    key_value = _noncontiguous_randn(
        *key_value_batch,
        key_value_len,
        channels,
        device=device,
        dtype=dtype,
    )
    attn_mask = torch.randint(
        0,
        2,
        (*query_batch, query_len, key_value_len),
        dtype=torch.bool,
        device=device,
    )
    attn_mask[..., 0] = True

    output_batch = torch.broadcast_shapes(query_batch, key_value_batch)
    exact_calls = None
    if len(query_batch) == 1 and query_batch == key_value_batch:
        exact_calls = (output_batch.numel() + limit - 1) // limit
    cache_tail = (
        key_value_len,
        num_key_value_heads,
        channels // num_query_heads,
    )

    with torch.no_grad():
        expected, expected_kv = module(
            query=query,
            key_value=key_value,
            attn_mask=attn_mask,
            rope=rope,
            return_key_value=True,
        )

        record_batch_sizes: list[int] = []
        handle = module.mlp.register_forward_hook(
            lambda _module, _inputs, output: record_batch_sizes.append(
                output.shape[:-2].numel()
            )
        )
        try:
            recorded, kv = module(
                query=query,
                key_value=key_value,
                attn_mask=attn_mask,
                rope=rope,
                batch_size_limit=limit,
                return_key_value=True,
            )
        finally:
            handle.remove()
        assert len(record_batch_sizes) > 1
        assert all(size <= limit for size in record_batch_sizes)
        if exact_calls is not None:
            assert len(record_batch_sizes) == exact_calls

        assert kv.key.shape == (*key_value_batch, *cache_tail)
        assert kv.value.shape == (*key_value_batch, *cache_tail)
        torch.testing.assert_close(kv.key, expected_kv.key)
        torch.testing.assert_close(kv.value, expected_kv.value)
        torch.testing.assert_close(recorded, expected)

        key_before = kv.key.clone()
        value_before = kv.value.clone()
        key_ptr = kv.key.data_ptr()
        value_ptr = kv.value.data_ptr()
        replay_batch_sizes: list[int] = []
        handle = module.mlp.register_forward_hook(
            lambda _module, _inputs, output: replay_batch_sizes.append(
                output.shape[:-2].numel()
            )
        )
        try:
            replayed = module(
                query=query,
                key_value=kv,
                attn_mask=attn_mask,
                rope=rope,
                batch_size_limit=limit,
            )
        finally:
            handle.remove()
        assert len(replay_batch_sizes) > 1
        assert all(size <= limit for size in replay_batch_sizes)
        if exact_calls is not None:
            assert len(replay_batch_sizes) == exact_calls

    assert replayed.shape == (*output_batch, query_len, channels)
    torch.testing.assert_close(replayed, expected)
    assert kv.key.data_ptr() == key_ptr
    assert kv.value.data_ptr() == value_ptr
    torch.testing.assert_close(kv.key, key_before)
    torch.testing.assert_close(kv.value, value_before)


@withCUDA
def test_transformer_block_kv_cache_batch_size_limit_guards(
    device: torch.device,
) -> None:
    module = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
        device=device,
    )
    _randomize_block_exits(module)
    query = torch.randn(4, 3, 8, device=device)
    key_value = torch.randn(4, 5, 8, device=device)
    with torch.no_grad():
        _, kv = module(
            query=query,
            key_value=key_value,
            return_key_value=True,
        )
        expected = module(query=query, key_value=kv)

    # Global grad mode, rather than `.training`, controls the inference-only
    # optimization. Cache replay must therefore remain a single block call.
    grad_calls: list[None] = []
    handle = module.mlp.register_forward_hook(
        lambda _module, _inputs, _output: grad_calls.append(None)
    )
    try:
        grad_enabled = module(
            query=query,
            key_value=kv,
            batch_size_limit=2,
        )
    finally:
        handle.remove()
    assert len(grad_calls) == 1
    torch.testing.assert_close(grad_enabled, expected)

    compile_calls: list[None] = []
    handle = module.mlp.register_forward_hook(
        lambda _module, _inputs, _output: compile_calls.append(None)
    )
    try:
        with (
            torch.no_grad(),
            mock.patch.object(
                torch.compiler, "is_compiling", return_value=True
            ),
        ):
            compiling = module(
                query=query,
                key_value=kv,
                batch_size_limit=2,
            )
    finally:
        handle.remove()
    assert len(compile_calls) == 1
    torch.testing.assert_close(compiling, expected)


def test_batch_size_limit_empty_query_cache_creation() -> None:
    query = torch.randn(1, 0, 8)
    key_value = torch.randn(5, 3, 8)
    modules: list[Attention | TransformerBlock] = [
        Attention(channels=8, num_query_heads=2),
        TransformerBlock(
            channels=8,
            num_query_heads=2,
            feedforward_channels=16,
        ),
    ]

    for module in modules:
        projector = (
            module.attn if isinstance(module, TransformerBlock) else module
        )
        with torch.no_grad():
            expected_out, expected_cache = module(
                query=query,
                key_value=key_value,
                return_key_value=True,
            )
            with mock.patch.object(
                projector,
                "_project_key_value",
                wraps=projector._project_key_value,
            ) as spy:
                actual_out, actual_cache = module(
                    query=query,
                    key_value=key_value,
                    return_key_value=True,
                    batch_size_limit=2,
                )
            limited_out = module(
                query=query,
                key_value=key_value,
                batch_size_limit=2,
            )

        assert spy.call_count == 3
        assert actual_out.shape == limited_out.shape == query.shape
        torch.testing.assert_close(actual_out, expected_out)
        torch.testing.assert_close(limited_out, expected_out)
        torch.testing.assert_close(actual_cache.key, expected_cache.key)
        torch.testing.assert_close(actual_cache.value, expected_cache.value)


def test_batch_size_limit_self_attention_cache_round_trip() -> None:
    query = torch.randn(5, 3, 8, dtype=torch.float64)
    attention = Attention(
        channels=8,
        num_query_heads=2,
        dtype=torch.float64,
    )
    block = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
        dtype=torch.float64,
    )
    _randomize_attention_exit(attention)
    _randomize_block_exits(block)

    for module in (attention, block):
        with torch.no_grad():
            expected, expected_cache = module(
                query=query,
                return_key_value=True,
            )
            recorded, cache = module(
                query=query,
                return_key_value=True,
                batch_size_limit=2,
            )
            replayed, replayed_cache = module(
                query=query,
                key_value=cache,
                return_key_value=True,
                batch_size_limit=2,
            )

        assert cache.key.shape == (5, 3, 2, 4)
        assert replayed_cache.key.data_ptr() == cache.key.data_ptr()
        assert replayed_cache.value.data_ptr() == cache.value.data_ptr()
        torch.testing.assert_close(cache.key, expected_cache.key)
        torch.testing.assert_close(cache.value, expected_cache.value)
        torch.testing.assert_close(recorded, expected)
        torch.testing.assert_close(replayed, expected)


@withCUDA
@pytest.mark.parametrize("batch_size_limit", [1, 3, 4])
@pytest.mark.parametrize("self_attn", [False, True])
def test_attention_batch_size_limit(
    device: torch.device,
    batch_size_limit: int,
    self_attn: bool,
) -> None:
    channels = 8
    num_heads = 2
    batch_size = 10
    # float64 makes the chunked-vs-unchunked comparison exact and immune to
    # kernel/TF32 differences from the varying per-chunk batch size.
    dtype = torch.float64
    module = Attention(
        channels=channels,
        num_query_heads=num_heads,
        device=device,
        dtype=dtype,
    )

    query = torch.randn(batch_size, 6, channels, device=device, dtype=dtype)
    key_value = (
        None
        if self_attn
        else torch.randn(batch_size, 5, channels, device=device, dtype=dtype)
    )
    kv_len = 6 if self_attn else 5
    # Keep column 0 unmasked so no query row is fully masked (avoids NaNs).
    attn_mask = torch.ones(
        batch_size, 6, kv_len, dtype=torch.bool, device=device
    )
    attn_mask[..., 1:] = torch.randint(
        0, 2, (batch_size, 6, kv_len - 1), dtype=torch.bool, device=device
    )

    # Count _attend calls to confirm the batch is actually chunked.
    with mock.patch.object(module, "_attend", wraps=module._attend) as spy:
        with torch.no_grad():
            chunked = module(
                query=query,
                key_value=key_value,
                attn_mask=attn_mask,
                batch_size_limit=batch_size_limit,
            )
        expected = (batch_size + batch_size_limit - 1) // batch_size_limit
        assert spy.call_count == expected

        # Gradients enabled -> chunking is bypassed (a single unchunked call).
        spy.reset_mock()
        unchunked = module(
            query=query,
            key_value=key_value,
            attn_mask=attn_mask,
            batch_size_limit=batch_size_limit,
        )
        assert spy.call_count == 1

    # Chunked inference output matches the unchunked computation exactly.
    torch.testing.assert_close(chunked, unchunked)


def test_attention_batch_size_limit_errors() -> None:
    query = torch.randn(4, 3, 8)
    mha = Attention(channels=8, num_query_heads=2)
    with pytest.raises(ValueError, match="must be positive"):
        mha(query=query, batch_size_limit=0)
    with pytest.raises(ValueError, match="must be positive"):
        mha(query=query, batch_size_limit=-1)
    block = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
    )
    with pytest.raises(ValueError, match="must be positive"):
        block(query=query, batch_size_limit=0)


def test_attention_batch_size_limit_nested_bypass() -> None:
    query = torch.nested.nested_tensor_from_jagged(
        values=torch.randn(5, 8),
        offsets=torch.tensor([0, 2, 5]),
    )
    attention = Attention(channels=8, num_query_heads=2)
    block = TransformerBlock(
        channels=8,
        num_query_heads=2,
        feedforward_channels=16,
    )

    with torch.no_grad():
        expected_attention = attention(query=query)
        expected_block = block(query=query)
        actual_attention = attention(query=query, batch_size_limit=1)
        actual_block = block(query=query, batch_size_limit=1)

    assert actual_attention.is_nested
    assert actual_block.is_nested
    torch.testing.assert_close(
        actual_attention.values(), expected_attention.values()
    )
    torch.testing.assert_close(actual_block.values(), expected_block.values())


@withCUDA
def test_transformer_block_batch_size_limit(device: torch.device) -> None:
    channels = 8
    num_heads = 2
    batch_size = 12
    limit = 4
    dtype = torch.float64
    module = TransformerBlock(
        channels=channels,
        num_query_heads=num_heads,
        feedforward_channels=16,
        device=device,
        dtype=dtype,
    )

    query = torch.randn(batch_size, 3, channels, device=device, dtype=dtype)
    key_value = torch.randn(
        batch_size, 5, channels, device=device, dtype=dtype
    )

    # Count _block calls to confirm the whole block is chunked.
    with mock.patch.object(module, "_block", wraps=module._block) as spy:
        with torch.no_grad():
            chunked = module(
                query=query, key_value=key_value, batch_size_limit=limit
            )
        assert spy.call_count == (batch_size + limit - 1) // limit

        # Gradients enabled -> whole-block chunking is bypassed (no _block
        # calls).
        spy.reset_mock()
        unchunked = module(
            query=query, key_value=key_value, batch_size_limit=limit
        )
        assert spy.call_count == 0

    torch.testing.assert_close(chunked, unchunked)


def test_attention_key_value_cache_dtype_mismatch() -> None:
    from sdm.cache import KVCacheEntry

    torch.manual_seed(0)
    module = Attention(channels=8, num_query_heads=2)
    query = torch.randn(2, 3, 8)
    cached = KVCacheEntry(
        key=torch.randn(2, 5, 2, 4, dtype=torch.bfloat16),
        value=torch.randn(2, 5, 2, 4, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="Re-run the caching step"):
        module(query=query, key_value=cached)

    # A value-only mismatch must be caught as well.
    cached = KVCacheEntry(
        key=torch.randn(2, 5, 2, 4),
        value=torch.randn(2, 5, 2, 4, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="Re-run the caching step"):
        module(query=query, key_value=cached)


@withCUDA
def test_attention_key_value_cache_autocast(device: torch.device) -> None:
    torch.manual_seed(0)
    if device.type != "cuda":
        pytest.skip("autocast('cuda') requires a GPU")
    module = Attention(channels=8, num_query_heads=2, device=device)
    query = torch.randn(2, 3, 8, device=device)
    key_value = torch.randn(2, 5, 8, device=device)

    # Caching and replaying under the same autocast context is valid even
    # though the pre-projection query stays float32.
    with torch.amp.autocast("cuda", torch.bfloat16):
        out, cached = module(
            query=query, key_value=key_value, return_key_value=True
        )
        replayed = module(query=query, key_value=cached)
    torch.testing.assert_close(replayed, out)
