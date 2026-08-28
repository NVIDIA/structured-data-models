import torch
from torch.nn import GELU, Linear, RMSNorm, Sequential

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import PerHeadLogNScale
from sdm.testing import withCUDA


@withCUDA
def test_transformer_block_initialization(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)
    assert torch.count_nonzero(block.attn.out_lin.weight) == 0
    assert block.attn.sdpa.scale is None
    assert block.attn.sdpa.query_scaling is None
    assert block.post_attn_norm is None
    mlp = block.mlp
    assert isinstance(mlp, Sequential)
    assert len(mlp) == 4
    assert isinstance(mlp[0], RMSNorm)
    assert isinstance(mlp[1], Linear)
    assert mlp[1].in_features == 32
    assert mlp[1].out_features == 64
    assert mlp[1].bias is not None
    assert isinstance(mlp[2], GELU)
    assert isinstance(mlp[3], Linear)
    assert mlp[3].in_features == 64
    assert mlp[3].out_features == 32
    assert mlp[3].bias is not None
    assert torch.count_nonzero(mlp[3].weight) == 0
    assert torch.count_nonzero(mlp[3].bias) == 0

    for transform in (block.attn.query_transform, block.attn.key_transform):
        assert isinstance(transform, Sequential)
        assert len(transform) == 1
        norm = transform[0]
        assert isinstance(norm, RMSNorm)
        assert norm.normalized_shape == (8,)
        assert norm.eps == 1e-6
        assert not norm.elementwise_affine
        assert norm.weight is None
        assert not list(transform.parameters())


@withCUDA
def test_transformer_block_per_head_logn_scale(
    device: torch.device,
) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        per_head_logn_scale=True,
        device=device,
    )

    assert isinstance(block.attn.sdpa.query_scaling, PerHeadLogNScale)
