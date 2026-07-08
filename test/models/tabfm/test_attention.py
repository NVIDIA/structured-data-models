from types import ModuleType

import pytest
import torch
from sdm.models.tabfm.attention import (
    Encoder,
    MultiheadAttention,
    MultiheadAttentionBlock,
    RMSNorm,
    RotaryEmbedding,
)


def test_rms_norm_matches_upstream(
    upstream_tabfm_module: ModuleType,
) -> None:
    upstream = upstream_tabfm_module.RMSNorm(dim=8, eps=1e-6)
    model = RMSNorm(channels=8, epsilon=1e-6)
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(2, 5, 8)

    torch.testing.assert_close(
        model(input),
        upstream(input),
        rtol=0,
        atol=0,
    )


def test_rotary_embedding_matches_upstream(
    upstream_tabfm_module: ModuleType,
) -> None:
    upstream = upstream_tabfm_module.RoPE(dim=8, base=100_000.0)
    model = RotaryEmbedding(channels=8, theta=100_000.0)
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(2, 7, 4, 8)

    torch.testing.assert_close(
        model(input),
        upstream.rotate(input),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_multihead_attention_matches_upstream(
    upstream_tabfm_module: ModuleType,
    use_rope: bool,
    use_mask: bool,
) -> None:
    rope_theta = 100_000.0 if use_rope else None
    upstream = upstream_tabfm_module.MultiheadAttention(
        d_model=16,
        nhead=4,
        rope_base=rope_theta,
    )
    model = MultiheadAttention(
        channels=16,
        num_heads=4,
        rope_theta=rope_theta,
    )
    model.load_state_dict(upstream.state_dict())

    query = torch.randn(2, 5, 16)
    key = torch.randn(2, 7, 16)
    value = torch.randn(2, 7, 16)
    mask = None
    if use_mask:
        mask = torch.rand(2, 1, 5, 7) > 0.25

    upstream_rope = None
    model_rope = None
    if use_rope:
        upstream_rope = upstream_tabfm_module.RoPE(
            dim=4,
            base=100_000.0,
        )
        model_rope = RotaryEmbedding(channels=4, theta=100_000.0)
        model_rope.load_state_dict(upstream_rope.state_dict())

    torch.testing.assert_close(
        model(
            query,
            key,
            value,
            attn_mask=mask,
            rope=model_rope,
        ),
        upstream(
            query,
            key,
            value,
            attn_mask=mask,
            rope=upstream_rope,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
def test_multihead_attention_block_matches_upstream(
    upstream_tabfm_module: ModuleType,
    activation: str,
) -> None:
    upstream = upstream_tabfm_module.MultiheadAttentionBlock(
        d_model=16,
        nhead=4,
        dim_ff=32,
        activation=activation,
        rope_base=100_000.0,
    )
    model = MultiheadAttentionBlock(
        channels=16,
        num_heads=4,
        feedforward_channels=32,
        activation=activation,
        rope_theta=100_000.0,
    )
    model.load_state_dict(upstream.state_dict())
    query = torch.randn(2, 5, 16)
    key_value = torch.randn(2, 7, 16)
    mask = torch.rand(2, 1, 5, 7) > 0.25
    upstream_rope = upstream_tabfm_module.RoPE(dim=4, base=100_000.0)
    model_rope = RotaryEmbedding(channels=4, theta=100_000.0)
    model_rope.load_state_dict(upstream_rope.state_dict())

    torch.testing.assert_close(
        model(
            query,
            key_value,
            key_value,
            attn_mask=mask,
            rope=model_rope,
        ),
        upstream(
            query,
            key_value,
            key_value,
            attn_mask=mask,
            rope=upstream_rope,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_feedforward_chunking_preserves_output() -> None:
    model = MultiheadAttentionBlock(
        channels=16,
        num_heads=4,
        feedforward_channels=32,
    ).eval()
    input = torch.randn(2, 7, 16)

    unchunked = model(input)
    model.ffn_chunk_size = 3
    chunked = model(input)

    torch.testing.assert_close(chunked, unchunked, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_encoder_matches_upstream(
    upstream_tabfm_module: ModuleType,
    use_rope: bool,
    dtype: torch.dtype,
) -> None:
    rope_theta = 100_000.0 if use_rope else None
    upstream = upstream_tabfm_module.Encoder(
        num_blocks=2,
        d_model=16,
        nhead=4,
        dim_ff=32,
        activation="swiglu",
        rope_base=rope_theta,
    ).to(dtype)
    model = Encoder(
        num_blocks=2,
        channels=16,
        num_heads=4,
        feedforward_channels=32,
        activation="swiglu",
        rope_theta=rope_theta,
    ).to(dtype)
    model.load_state_dict(upstream.state_dict())
    input = torch.randn(2, 7, 16, dtype=dtype)
    mask = torch.rand(2, 1, 7, 7) > 0.25

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)

    torch.testing.assert_close(
        model(input, attn_mask=mask),
        upstream(input, attn_mask=mask),
        rtol=rtol,
        atol=atol,
    )
