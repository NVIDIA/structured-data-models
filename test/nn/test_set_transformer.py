import pytest
import torch

from sdm.nn import InducedTransformerBlock, TransformerBlock
from sdm.testing import withCUDA


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
def test_induced_transformer_block_transformer_2_float32(
    device: torch.device,
) -> None:
    output_mlp = torch.nn.Sequential(
        torch.nn.Linear(8, 16, device=device),
        torch.nn.GELU(),
        torch.nn.Linear(16, 8, device=device),
    )
    module = InducedTransformerBlock(
        channels=8,
        num_inducing_points=4,
        inducing_block=TransformerBlock(
            channels=8,
            num_query_heads=2,
            mlp=torch.nn.Identity(),
            device=device,
        ),
        output_block=TransformerBlock(
            channels=8,
            num_query_heads=2,
            mlp=output_mlp,
            device=device,
        ),
        device=device,
    )
    query = torch.randn(2, 6, 8, device=device)
    key_value = torch.randn(2, 5, 8, device=device)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    torch.nn.init.constant_(module.output_block.attn.out_lin.weight, 0.02)
    mlp_out = module.output_block.mlp[-1]
    assert isinstance(mlp_out, torch.nn.Linear)
    torch.nn.init.constant_(mlp_out.weight, 0.02)

    with torch.amp.autocast(device.type, dtype=autocast_dtype):
        induced = module.inducing_block(
            query=module.inducing_points,
            key_value=key_value,
        )
    with torch.amp.autocast(device.type, enabled=False):
        expected_out, expected_kv = module.output_block(
            query=query.float(),
            key_value=induced.float(),
            return_key_value=True,
        )
    with torch.amp.autocast(device.type, dtype=autocast_dtype):
        out, kv = module(
            query=query,
            key_value=key_value,
            return_key_value=True,
            _transformer_2_float32=True,
        )

    assert out.dtype == torch.float32
    assert kv.key.dtype == torch.float32
    assert kv.value.dtype == torch.float32
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(kv.key, expected_kv.key)
    torch.testing.assert_close(kv.value, expected_kv.value)
