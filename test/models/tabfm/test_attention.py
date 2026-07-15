import math

import pytest
import torch
import torch.nn.functional as F
from sdm.models.tabfm.attention import MultiheadAttention
from sdm.nn import RotaryEmbedding
from sdm.testing import withCUDA
from torch import Tensor
from torch.nn import RMSNorm


def _rms_norm(x: Tensor, weight: Tensor, epsilon: float = 1e-6) -> Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    return (x * (variance + epsilon).rsqrt() * weight.float()).to(dtype)


def _reference_attention(
    module: MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    rope: RotaryEmbedding | None,
) -> Tensor:
    batch_size, query_length, channels = query.shape
    key_length = key.size(1)
    query = F.linear(query, module.q_proj.weight, module.q_proj.bias).view(
        batch_size,
        query_length,
        module.num_heads,
        module.head_channels,
    )
    key = F.linear(key, module.k_proj.weight, module.k_proj.bias).view(
        batch_size,
        key_length,
        module.num_heads,
        module.head_channels,
    )
    value = F.linear(value, module.v_proj.weight, module.v_proj.bias).view(
        batch_size,
        key_length,
        module.num_heads,
        module.head_channels,
    )
    if rope is not None:
        query = rope(query)
        key = rope(key)

    query = _rms_norm(query, module.query_ln.weight)
    key = _rms_norm(key, module.key_ln.weight)
    scale = (
        1.442695041
        / math.sqrt(module.head_channels)
        * F.softplus(module.per_dim_scale.float())
    )
    query = query * scale.to(query.dtype)
    output = F.scaled_dot_product_attention(
        query=query.transpose(1, 2),
        key=key.transpose(1, 2),
        value=value.transpose(1, 2),
        attn_mask=attn_mask,
        scale=1.0,
    )
    output = output.transpose(1, 2).reshape(
        batch_size,
        query_length,
        channels,
    )
    return F.linear(output, module.out_proj.weight, module.out_proj.bias)


@withCUDA
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_multihead_attention_matches_tabfm_reference(
    device: torch.device,
    use_rope: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    module = MultiheadAttention(
        channels=8,
        num_heads=2,
        device=device,
        dtype=dtype,
    )
    rope = (
        RotaryEmbedding(
            channels=4,
            layout="interleaved",
            requires_grad=False,
            device=device,
            dtype=dtype,
        )
        if use_rope
        else None
    )
    query = torch.randn(2, 3, 8, device=device, dtype=dtype)
    key = torch.randn(2, 5, 8, device=device, dtype=dtype)
    value = torch.randn(2, 5, 8, device=device, dtype=dtype)
    attn_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]],
        device=device,
    )[:, None, None, :]

    output = module(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        rope=rope,
    )
    expected = _reference_attention(
        module=module,
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        rope=rope,
    )

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    assert output.shape == query.shape
    assert output.device == device
    assert output.dtype == query.dtype


def test_multihead_attention_uses_built_in_primitives() -> None:
    module = MultiheadAttention(channels=8, num_heads=2)

    assert isinstance(module.query_ln, RMSNorm)
    assert isinstance(module.key_ln, RMSNorm)
    assert module.query_ln.eps == 1e-6
    assert module.key_ln.eps == 1e-6
    assert module.sdpa.scale == 1.0

    expected_scale = torch.full((4,), 1 / math.sqrt(4))
    actual_scale = (
        1.442695041 / math.sqrt(4) * F.softplus(module.per_dim_scale.float())
    )
    torch.testing.assert_close(actual_scale, expected_scale)
    assert set(module.state_dict()) == {
        "per_dim_scale",
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
        "out_proj.weight",
        "out_proj.bias",
        "query_ln.weight",
        "key_ln.weight",
    }


@pytest.mark.parametrize(
    ("channels", "num_heads", "match"),
    [(0, 1, "channels"), (8, 3, "num_heads")],
)
def test_multihead_attention_rejects_invalid_configuration(
    channels: int,
    num_heads: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MultiheadAttention(channels=channels, num_heads=num_heads)


def test_multihead_attention_rejects_incompatible_inputs() -> None:
    module = MultiheadAttention(channels=8, num_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="batch dimensions"):
        module(query, key[:1], value)
    with pytest.raises(ValueError, match="sequence lengths"):
        module(query, key, value[:, :-1])
    with pytest.raises(ValueError, match="channels"):
        module(query, key, torch.randn(2, 5, 7))
