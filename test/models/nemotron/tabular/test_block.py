import pytest
import torch
from torch.nn import RMSNorm, Sequential

from sdm.models.nemotron.tabular.block import _TransformerBlock
from sdm.nn import RotaryEmbedding, SwiGLU
from sdm.testing import withCUDA


@pytest.mark.parametrize(
    ("channels", "num_heads", "expected_num_parameters"),
    [(128, 8, 165_040), (512, 8, 2_626_240)],
)
def test_transformer_block_parameter_count(
    channels: int,
    num_heads: int,
    expected_num_parameters: int,
) -> None:
    block = _TransformerBlock(channels=channels, num_heads=num_heads)

    assert sum(parameter.numel() for parameter in block.parameters()) == (
        expected_num_parameters
    )


@withCUDA
def test_transformer_block_initialization(device: torch.device) -> None:
    block = _TransformerBlock(channels=32, num_heads=4, device=device)
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)
    assert torch.count_nonzero(block.attn.out_lin.weight) > 0
    mlp = block.mlp
    assert isinstance(mlp, Sequential)
    swiglu = mlp[1]
    assert isinstance(swiglu, SwiGLU)
    assert torch.count_nonzero(swiglu.down_lin.weight) > 0
    post_attn_norm = block.post_attn_norm
    assert isinstance(post_attn_norm, RMSNorm)
    assert torch.count_nonzero(post_attn_norm.weight) == 0
    post_mlp_norm = mlp[-1]
    assert isinstance(post_mlp_norm, RMSNorm)
    assert torch.count_nonzero(post_mlp_norm.weight) == 0


@withCUDA
def test_transformer_block_rope(device: torch.device) -> None:
    rope = RotaryEmbedding(
        channels=16,
        layout="split_half",
        requires_grad=False,
        partial_rotary_factor=0.25,
        device=device,
    )
    block = _TransformerBlock(
        channels=128,
        num_heads=8,
        rope=rope,
        device=device,
    )
    query = torch.randn(2, 5, 128, device=device)
    post_attn_norm = block.post_attn_norm
    assert isinstance(post_attn_norm, RMSNorm)

    with torch.no_grad():
        post_attn_norm.weight.fill_(1)
        rope.inv_freq.zero_()
        output_without_rotation = block(query)
        rope.inv_freq.fill_(torch.pi / 2)
        output = block(query)

    assert output.shape == query.shape
    assert output.device == device
    assert not torch.allclose(output, output_without_rotation)
