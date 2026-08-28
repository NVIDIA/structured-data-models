import pytest
import torch
from torch.nn import RMSNorm, Sequential

from sdm.cache import Cache
from sdm.models.kumo.tabular.table_encoder import TableEncoder
from sdm.nn import PerHeadLogNScale, RotaryEmbedding
from sdm.testing import withCUDA


def test_requires_stage() -> None:
    with pytest.raises(ValueError, match="'num_stages' must be at least 1"):
        TableEncoder(num_stages=0)


def test_stage_configuration() -> None:
    encoder = TableEncoder(
        channels=128,
        num_inducing_points=4,
        num_stages=2,
    )

    assert isinstance(encoder.norm, RMSNorm)
    assert encoder.norm.normalized_shape == (128,)
    assert not hasattr(encoder, "col_projections")
    assert not hasattr(encoder, "row_norms")
    for col_block, row_block in zip(encoder.col_blocks, encoder.row_blocks):
        inducing_attn = col_block.inducing_block.attn
        output_attn = col_block.output_block.attn
        row_attn = row_block.attn

        assert inducing_attn.num_query_heads == 4
        assert output_attn.num_query_heads == 4
        assert row_attn.num_query_heads == 4
        assert isinstance(
            inducing_attn.sdpa.query_scaling,
            PerHeadLogNScale,
        )
        assert inducing_attn.sdpa.query_scaling.head_scale.shape == (4,)
        assert output_attn.sdpa.query_scaling is None
        assert row_attn.sdpa.query_scaling is None

        query_transform = row_attn.query_transform
        key_transform = row_attn.key_transform
        assert isinstance(query_transform, Sequential)
        assert isinstance(key_transform, Sequential)
        query_rope = query_transform[0]
        key_rope = key_transform[0]
        assert isinstance(query_rope, RotaryEmbedding)
        assert isinstance(key_rope, RotaryEmbedding)
        assert query_rope is key_rope
        assert query_rope.layout == "split_half"
        assert query_rope.channels == 32
        assert query_rope.rotary_channels == 8
        assert query_rope.inv_freq.shape == (4,)
        assert not query_rope.inv_freq.requires_grad
        expected_inv_freq = 1.0 / (100_000 ** (torch.arange(0, 8, 2) / 8))
        torch.testing.assert_close(query_rope.inv_freq, expected_inv_freq)

        for transform in (query_transform, key_transform):
            assert len(transform) == 2
            norm = transform[1]
            assert isinstance(norm, RMSNorm)
            assert norm.normalized_shape == (32,)
            assert norm.eps == 1e-6
            assert not norm.elementwise_affine
            assert norm.weight is None


@withCUDA
def test_table_encoder(device: torch.device) -> None:
    encoder = TableEncoder(
        channels=16,
        num_heads=2,
        num_inducing_points=4,
        num_cls_tokens=2,
        device=device,
    )
    context = torch.randn(2, 3, 3, 16, device=device)
    query = torch.randn(2, 2, 3, 16, device=device)

    out = encoder(
        torch.cat((context, query), dim=-3),
        num_context_rows=context.size(-3),
    )
    assert out.size() == (2, 5, 32)
    assert out.device == device

    cache = Cache()
    encoder(context, num_context_rows=context.size(-3), cache=cache)
    out = encoder(query, num_context_rows=0, cache=cache.freeze())
    assert out.size() == (2, 2, 32)
    assert out.device == device

    with torch.no_grad():
        encoder.norm.weight.zero_()
    out = encoder(
        torch.cat((context, query), dim=-3),
        num_context_rows=context.size(-3),
    )
    assert torch.count_nonzero(out) == 0
