import pytest
import torch
from sdm.nn import InducedTransformerBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("qassmax", [False, True])
def test_induced_transformer_block(
    device: torch.device,
    qassmax: bool,
) -> None:
    batch_size = 2
    set_size = 6
    channels = 8
    module = InducedTransformerBlock(
        channels=channels,
        num_heads=2,
        feedforward_channels=16,
        num_inducing_points=4,
        qassmax=qassmax,
        device=device,
    )
    query = torch.randn(batch_size, set_size, channels, device=device)

    # Self-attention: `key_value` defaults to `query`.
    out = module(query)
    assert out.shape == query.shape
    assert out.dtype == query.dtype
    assert out.device == query.device

    # Cross-attention against a separate key/value context.
    context_size = 4
    key_value = torch.randn(batch_size, context_size, channels, device=device)
    out_cross = module(query, key_value=key_value)
    assert out_cross.shape == query.shape

    # Passing `key_value=query` explicitly matches the self-attention default.
    out_explicit = module(query, key_value=query)
    torch.testing.assert_close(out_explicit, out)

    # Per-batch valid context lengths restrict the attended key/value range.
    seqused = torch.full(
        (batch_size,), context_size, dtype=torch.int32, device=device
    )
    out_seqused = module(query, key_value=key_value, seqused_key_value=seqused)
    torch.testing.assert_close(out_seqused, out_cross)


@pytest.mark.parametrize("qassmax", [False, True])
def test_induced_transformer_block_kv_cache(qassmax: bool) -> None:
    batch_size = 2
    set_size = 6
    context_size = 5
    channels = 8
    num_heads = 2
    num_inducing_points = 4
    module = InducedTransformerBlock(
        channels=channels,
        num_heads=num_heads,
        feedforward_channels=16,
        num_inducing_points=num_inducing_points,
        qassmax=qassmax,
    )

    query = torch.randn(batch_size, set_size, channels)
    key_value = torch.randn(batch_size, context_size, channels)
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
    cached_out = module(query=query, key_value=kv)
    assert kv.key.size() == (
        batch_size,
        num_inducing_points,
        num_heads,
        channels // num_heads,
    )
    assert kv.value.size() == kv.key.size()
    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(cached_out, direct_out)

    # The cache encodes only the context, so an unrelated query reuses it and
    # matches a full forward of that query over the same context.
    other_query = torch.randn(batch_size, set_size, channels)
    other_direct_out = module(
        query=other_query,
        key_value=key_value,
        seqused_key_value=seqused_key_value,
    )
    other_cached_out = module(query=other_query, key_value=kv)
    torch.testing.assert_close(other_cached_out, other_direct_out)

    # Self-attention reuses the same cached key/value path.
    self_out = module(query=query)
    self_cache_out, self_kv = module(query=query, return_key_value=True)
    self_cached_out = module(query=query, key_value=self_kv)
    assert self_kv.key.size() == (
        batch_size,
        num_inducing_points,
        num_heads,
        channels // num_heads,
    )
    assert self_kv.value.size() == self_kv.key.size()
    torch.testing.assert_close(self_cache_out, self_out)
    torch.testing.assert_close(self_cached_out, self_out)
