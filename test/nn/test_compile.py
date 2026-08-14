import os
from collections.abc import Callable, Iterator

import pytest
import torch
from torch import Tensor

from sdm.nn import (
    SDPA,
    Attention,
    InducedTransformerBlock,
    QASSMax,
    RotaryEmbedding,
    TransformerBlock,
)
from sdm.testing import withCUDA

# Skip all tests in this test file if it is not a full test run (FULL_TEST=1).
pytestmark = pytest.mark.skipif(
    os.getenv("FULL_TEST", "0") != "1",
    reason="Fast test run",
)


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    # Isolate compile caches so per-test guard specializations of shared
    # forward code objects never hit the recompile limit.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def fullgraph(module: torch.nn.Module) -> Callable[..., Tensor]:
    # `fullgraph=True` raises on any graph break; the eager backend skips
    # code generation, keeping the tests fast while staying numerically
    # identical to the uncompiled module.
    return torch.compile(module, fullgraph=True, backend="eager")


@withCUDA
@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1], [2]]),
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
    ],
)
def test_qassmax_compile(
    device: torch.device,
    key_len_fn: Callable[[], Tensor | int],
) -> None:
    module = QASSMax(channels=2, num_heads=3, hidden_channels=4, device=device)
    query = torch.randn(2, 3, 3, 2, device=device)
    key_len = key_len_fn()
    if isinstance(key_len, Tensor):
        key_len = key_len.to(device)

    expected = module(query, key_len=key_len)
    out = fullgraph(module)(query, key_len=key_len)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_rotary_embedding_compile(device: torch.device) -> None:
    module = RotaryEmbedding(channels=4, layout="split_half", device=device)
    x = torch.randn(2, 5, 3, 4, device=device)

    expected = module(x)
    out = fullgraph(module)(x)
    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize(
    ("num_key_value_heads", "qassmax", "masking"),
    [
        (None, False, None),
        (None, True, "attn_mask"),
        (None, False, "seqused"),
        (1, False, None),
        (2, True, "attn_mask"),
    ],
)
def test_sdpa_compile(
    device: torch.device,
    num_key_value_heads: int | None,
    qassmax: bool,
    masking: str | None,
) -> None:
    channels = 4
    num_query_heads = 4
    module = SDPA(
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        query_scaling=QASSMax(
            channels // num_query_heads,
            num_query_heads,
            device=device,
        )
        if qassmax
        else None,
    )

    kv_heads = num_key_value_heads or num_query_heads
    query = torch.randn(2, 3, num_query_heads, channels, device=device)
    key = torch.randn(2, 5, kv_heads, channels, device=device)
    value = torch.randn(2, 5, kv_heads, channels, device=device)
    attn_mask = seqused = None
    if masking == "attn_mask":
        attn_mask = torch.ones(2, 3, 5, dtype=torch.bool, device=device)
        attn_mask[..., 1:] = torch.randint(
            0, 2, (2, 3, 4), dtype=torch.bool, device=device
        )
    elif masking == "seqused":
        seqused = torch.tensor([3, 1], dtype=torch.int32, device=device)

    expected = module(
        query=query,
        key=key,
        value=value,
        seqused_key_value=seqused,
        attn_mask=attn_mask,
    )
    out = fullgraph(module)(
        query=query,
        key=key,
        value=value,
        seqused_key_value=seqused,
        attn_mask=attn_mask,
    )
    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize(
    ("num_key_value_heads", "qassmax", "self_attn"),
    [
        (None, False, True),
        (None, True, False),
        (1, False, True),
        (2, True, False),
    ],
)
def test_attention_compile(
    device: torch.device,
    num_key_value_heads: int | None,
    qassmax: bool,
    self_attn: bool,
) -> None:
    channels = 8
    module = Attention(
        channels=channels,
        num_query_heads=4,
        num_key_value_heads=num_key_value_heads,
        query_scaling=QASSMax(channels // 4, num_heads=4, device=device)
        if qassmax
        else None,
        device=device,
    )
    query = torch.randn(2, 3, channels, device=device)
    key_value = (
        None if self_attn else torch.randn(2, 5, channels, device=device)
    )

    expected = module(query=query, key_value=key_value)
    out = fullgraph(module)(query=query, key_value=key_value)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_attention_compile_key_value_cache(device: torch.device) -> None:
    module = Attention(channels=8, num_query_heads=2, device=device)
    query = torch.randn(2, 3, 8, device=device)
    key_value = torch.randn(2, 5, 8, device=device)

    expected, expected_kv = module(
        query=query, key_value=key_value, return_key_value=True
    )
    out, kv = fullgraph(module)(
        query=query, key_value=key_value, return_key_value=True
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(kv.key, expected_kv.key)
    torch.testing.assert_close(kv.value, expected_kv.value)

    # Replaying the cached projections also compiles without graph breaks.
    expected = module(query=query, key_value=expected_kv)
    out = fullgraph(module)(query=query, key_value=kv)
    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize(
    ("qassmax", "masking"),
    [
        (False, None),
        (True, None),
        (False, "seqused"),
        (True, "attn_mask"),
    ],
)
def test_transformer_block_compile(
    device: torch.device,
    qassmax: bool,
    masking: str | None,
) -> None:
    channels = 8
    module = TransformerBlock(
        channels=channels,
        num_query_heads=2,
        feedforward_channels=16,
        query_scaling=QASSMax(channels // 2, num_heads=2, device=device)
        if qassmax
        else None,
        device=device,
    )
    query = torch.randn(2, 3, channels, device=device)
    key_value = torch.randn(2, 5, channels, device=device)
    attn_mask = seqused = None
    if masking == "attn_mask":
        attn_mask = torch.ones(2, 3, 5, dtype=torch.bool, device=device)
        attn_mask[..., 1:] = torch.randint(
            0, 2, (2, 3, 4), dtype=torch.bool, device=device
        )
    elif masking == "seqused":
        seqused = torch.tensor([3, 1], dtype=torch.int32, device=device)

    expected = module(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused,
        attn_mask=attn_mask,
    )
    out = fullgraph(module)(
        query=query,
        key_value=key_value,
        seqused_key_value=seqused,
        attn_mask=attn_mask,
    )
    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize("self_attn", [False, True])
def test_induced_transformer_block_compile(
    device: torch.device,
    self_attn: bool,
) -> None:
    channels = 8
    module = InducedTransformerBlock(
        channels=channels,
        num_query_heads=2,
        feedforward_channels=16,
        num_inducing_points=4,
        query_scaling=QASSMax(channels // 2, num_heads=2, device=device),
        device=device,
    )
    query = torch.randn(2, 6, channels, device=device)
    key_value = (
        None if self_attn else torch.randn(2, 5, channels, device=device)
    )

    expected = module(query=query, key_value=key_value)
    out = fullgraph(module)(query=query, key_value=key_value)
    torch.testing.assert_close(out, expected)
