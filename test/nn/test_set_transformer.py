import pytest
import torch

from sdm.nn import InducedTransformerBlock, TransformerBlock
from sdm.testing import withCUDA


def _randomize_residual_exits(module: InducedTransformerBlock) -> None:
    # The production modules intentionally start as residual identities. Make
    # attention and MLP outputs non-zero so equivalence exercises cache data.
    with torch.no_grad():
        for block in (module.inducing_block, module.output_block):
            block.attn.out_lin.weight.normal_(std=0.05)
            block.attn.out_lin.bias.normal_(std=0.05)
            # `mlp` is caller-injected, so find its last Linear (if any)
            # rather than assuming a Sequential.
            linears = [
                m
                for m in block.mlp.modules()
                if isinstance(m, torch.nn.Linear)
            ]
            if linears:
                linears[-1].weight.normal_(std=0.05)
                linears[-1].bias.normal_(std=0.05)


@withCUDA
@pytest.mark.parametrize("num_key_value_heads", [None, 1])
def test_induced_transformer_block(
    device: torch.device,
    num_key_value_heads: int | None,
) -> None:
    batch_size = 2
    set_size = 6
    channels = 8
    module = InducedTransformerBlock(
        channels=channels,
        num_inducing_points=4,
        inducing_block=TransformerBlock(
            channels=channels,
            num_query_heads=2,
            num_key_value_heads=num_key_value_heads,
            mlp=torch.nn.Identity(),
            device=device,
        ),
        output_block=TransformerBlock(
            channels=channels,
            num_query_heads=2,
            num_key_value_heads=num_key_value_heads,
            mlp=torch.nn.Identity(),
            device=device,
        ),
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


def test_induced_transformer_block_kv_cache() -> None:
    batch_size = 2
    set_size = 6
    context_size = 5
    channels = 8
    num_heads = 2
    num_inducing_points = 4
    module = InducedTransformerBlock(
        channels=channels,
        num_inducing_points=num_inducing_points,
        inducing_block=TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            mlp=torch.nn.Identity(),
        ),
        output_block=TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            mlp=torch.nn.Identity(),
        ),
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
    with torch.no_grad():
        chunked_out = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            batch_size_limit=1,
        )
        chunked_cache_out, chunked_kv = module(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            return_key_value=True,
            batch_size_limit=1,
        )
        chunked_cached_out = module(
            query=query,
            key_value=chunked_kv,
            batch_size_limit=1,
        )
    assert kv.key.size() == (
        batch_size,
        num_inducing_points,
        num_heads,
        channels // num_heads,
    )
    assert kv.value.size() == kv.key.size()
    torch.testing.assert_close(cache_out, direct_out)
    torch.testing.assert_close(cached_out, direct_out)
    torch.testing.assert_close(chunked_out, direct_out)
    torch.testing.assert_close(chunked_cache_out, direct_out)
    torch.testing.assert_close(chunked_cached_out, direct_out)
    torch.testing.assert_close(chunked_kv.key, kv.key)
    torch.testing.assert_close(chunked_kv.value, kv.value)

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


@withCUDA
def test_induced_transformer_block_batch_size_limit(
    device: torch.device,
) -> None:
    # Exercises batch chunking through the induced block, including the
    # broadcast of the unbatched inducing-point query against the batched
    # key/value inside transformer_1. Chunking only runs with gradients
    # disabled.
    batch_shape = (2, 6)
    batch_size_limit = 4
    set_size = 5
    context_size = 7
    channels = 8
    num_heads = 2
    num_inducing_points = 3
    dtype = torch.float64
    torch.manual_seed(0)
    module = InducedTransformerBlock(
        channels=channels,
        num_inducing_points=num_inducing_points,
        inducing_block=TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            mlp=torch.nn.Linear(
                channels, channels, device=device, dtype=dtype
            ),
            device=device,
            dtype=dtype,
        ),
        output_block=TransformerBlock(
            channels=channels,
            num_query_heads=num_heads,
            mlp=torch.nn.Linear(
                channels, channels, device=device, dtype=dtype
            ),
            device=device,
            dtype=dtype,
        ),
        device=device,
        dtype=dtype,
    )
    _randomize_residual_exits(module)

    query = torch.randn(
        *batch_shape, set_size, channels, device=device, dtype=dtype
    )
    key_value = torch.randn(
        *batch_shape, context_size, channels, device=device, dtype=dtype
    )

    with torch.no_grad():
        unchunked = module(query=query, key_value=key_value)
        chunked = module(
            query=query,
            key_value=key_value,
            batch_size_limit=batch_size_limit,
        )
        cache_out, kv = module(
            query=query,
            key_value=key_value,
            batch_size_limit=batch_size_limit,
            return_key_value=True,
        )

    torch.testing.assert_close(chunked, unchunked)
    torch.testing.assert_close(cache_out, unchunked)
    expected_cache_shape = (
        *batch_shape,
        num_inducing_points,
        num_heads,
        channels // num_heads,
    )
    assert kv.key.size() == expected_cache_shape
    assert kv.value.size() == expected_cache_shape

    # Cached replay skips transformer_1 and stays equivalent under a batch
    # limit as well.
    other_query = torch.randn_like(query)
    with torch.no_grad():
        other_unchunked = module(query=other_query, key_value=key_value)
        other_cached = module(
            query=other_query,
            key_value=kv,
            batch_size_limit=batch_size_limit,
        )

    torch.testing.assert_close(other_cached, other_unchunked)
