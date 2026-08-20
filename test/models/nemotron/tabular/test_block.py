import torch
from torch.nn import RMSNorm, Sequential

from sdm.models.nemotron.tabular.block import NemotronTabularTransformerBlock
from sdm.nn import SwiGLU
from sdm.testing import withCUDA


@withCUDA
def test_transformer_block_initialization(device: torch.device) -> None:
    block = NemotronTabularTransformerBlock(
        channels=32,
        num_heads=4,
        device=device,
    )
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
