from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F
from schemafm.nn.attention import SDPA, QASSMax
from torch import Tensor


def reference_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
) -> Tensor:
    return F.scaled_dot_product_attention(
        query=query.transpose(-3, -2),
        key=key.transpose(-3, -2),
        value=value.transpose(-3, -2),
        attn_mask=attn_mask.unsqueeze(-3) if attn_mask is not None else None,
    ).transpose(-3, -2)


@pytest.mark.parametrize(
    "key_len_fn",
    [
        lambda: 4,
        lambda: torch.tensor([[1, 2, 0], [3, 4, 1]]),
        lambda: torch.tensor([[4], [1]], dtype=torch.int32),
    ],
)
def test_qassmax(key_len_fn: Callable[[], Tensor | int]) -> None:
    channels = 2
    num_heads = 3
    module = QASSMax(channels=channels, num_heads=num_heads, hidden_channels=4)

    query = torch.randn(2, 3, num_heads, channels)

    out = module(query, key_len_fn())
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device


def test_sdpa() -> None:
    channels = 3
    num_heads = 2
    module = SDPA(channels=channels, num_heads=num_heads)

    # Match torch SDPA for unbatched query, key, and value tensors.
    query = torch.randn(4, num_heads, channels)
    key = torch.randn(5, num_heads, channels)
    value = torch.randn(5, num_heads, channels)

    out = module(query=query, key=key, value=value)
    expected = reference_sdpa(query=query, key=key, value=value)
    torch.testing.assert_close(out, expected)

    # Broadcast batch dimensions and apply a boolean attention mask.
    query = torch.randn(2, 3, num_heads, channels)
    key = torch.randn(5, num_heads, channels)
    value = torch.randn(1, 5, num_heads, channels)
    attn_mask = torch.randint(0, 2, (3, 5), dtype=torch.bool)

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
    query = torch.randn(batch_size, query_len, num_heads, channels)
    key = torch.randn(batch_size, key_value_len, num_heads, channels)
    value = torch.randn(batch_size, key_value_len, num_heads, channels)
    value[0, 3:] = 1000
    value[1, 1:] = -1000
    seqused_key_value = torch.tensor([3, 1], dtype=torch.int32)

    out = module(
        query=query,
        key=key,
        value=value,
        seqused_key_value=seqused_key_value,
    )

    key_index = torch.arange(key_value_len).view(1, 1, key_value_len)
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
        num_heads,
        channels,
    )
    key = torch.randn(batch_size, 1, num_train, num_heads, channels)
    value = torch.randn(batch_size, 1, num_train, num_heads, channels)

    out = module(query=query, key=key, value=value)
    expected = reference_sdpa(
        query=query,
        key=key.expand(-1, num_test, -1, -1, -1),
        value=value.expand(-1, num_test, -1, -1, -1),
    )
    torch.testing.assert_close(out, expected)

    # Reject invalid mask and sequence-length combinations.
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
