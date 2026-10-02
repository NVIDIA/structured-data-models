# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import pytest
import torch

from sdm.models.timesfm3.configs import TransformerConfig
from sdm.models.timesfm3.transformer import (
    MixingTransformer,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
    make_attn_mask,
)
from sdm.testing import withCUDA


@withCUDA
def test_make_attn_mask(device: torch.device) -> None:
    patch_mask = torch.tensor(
        [[True, False, False], [False, True, False]], device=device
    )
    mask = make_attn_mask(patch_mask)
    causal = torch.ones(3, 3, dtype=torch.bool, device=device).tril()
    expected = causal[None, None] & ~patch_mask[:, None, None, :]
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_attn_mask_noncausal(device: torch.device) -> None:
    patch_mask = torch.tensor([[True, False, False]], device=device)
    mask = make_attn_mask(patch_mask, causal=False)
    torch.testing.assert_close(mask, ~patch_mask[:, None, None, :])


@withCUDA
@pytest.mark.parametrize("rank", [3, 4])
def test_rotary_positional_embedding(
    device: torch.device,
    rank: int,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    shape = (2, 3, 4) if rank == 3 else (2, 3, 2, 4)
    inputs = torch.arange(
        math.prod(shape),
        device=device,
        dtype=torch.float32,
    ).reshape(shape)
    output = rope(inputs)

    torch.testing.assert_close(
        output.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
    )
    torch.testing.assert_close(output[0, 0], inputs[0, 0])
    assert output.shape == inputs.shape
    assert output.device == device


def test_rotary_positional_embedding_excludes_buffer_from_checkpoint() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    assert rope.state_dict() == {}


def test_rotary_positional_embedding_rejects_wrong_dimension() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    with pytest.raises(ValueError, match="must match the hidden dimension"):
        rope(torch.zeros(1, 2, 6))


def test_rotary_positional_embedding_rejects_wrong_rank() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    with pytest.raises(ValueError, match="rank 3 or 4"):
        rope(torch.zeros(2, 4))


def test_rotary_positional_embedding_meta_device() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device="meta")

    assert rope.timescale.device.type == "meta"


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_positional_embedding_rotation(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    inputs = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
        device=device,
        dtype=dtype,
    )

    output = rope(inputs)

    expected = torch.tensor(
        [math.cos(1.0), math.cos(0.01), math.sin(1.0), math.sin(0.01)],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(output[0, 1], expected)


@withCUDA
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_multi_head_attention_bfloat16(
    device: torch.device,
    use_sdpa: bool,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=use_sdpa,
        device=device,
        dtype=torch.bfloat16,
    )
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.bfloat16)

    output = attention(inputs)[0]

    assert output.dtype == torch.bfloat16
    assert output.isfinite().all()


@withCUDA
def test_multi_head_attention_sdpa_fully_masked(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        device=device,
        dtype=torch.float16,
        use_sdpa=True,
    )
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.float16)
    patch_mask = torch.ones(2, 3, device=device, dtype=torch.bool)

    output = attention(inputs, patch_mask=patch_mask)[0]

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_transformer_config_attention_defaults() -> None:
    transformer = TransformerConfig(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
    )

    assert transformer.use_memory_efficient_attention is True
    assert transformer.use_sdpa is True


def _transformer_config(
    *,
    use_memory_efficient_attention: bool = True,
    use_sdpa: bool = True,
) -> TransformerConfig:
    return TransformerConfig(
        model_dims=8,
        hidden_dims=12,
        num_heads=2,
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        use_memory_efficient_attention=use_memory_efficient_attention,
        use_sdpa=use_sdpa,
    )


@withCUDA
@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize(
    ("use_memory_efficient_attention", "expected"),
    [(True, 2.3272503), (False, 2.2684078)],
)
def test_mixing_transformer_uses_attention_scaling_config(
    device: torch.device,
    use_sdpa: bool,
    use_memory_efficient_attention: bool,
    expected: float,
) -> None:
    config = replace(
        _transformer_config(
            use_memory_efficient_attention=use_memory_efficient_attention,
            use_sdpa=use_sdpa,
        ),
        model_dims=2,
        num_heads=1,
        qk_norm="none",
        use_rope_seq=False,
    )
    transformer = MixingTransformer(
        config, use_variate_attention=False, device=device
    )
    with torch.no_grad():
        transformer.pre_seq_attn_ln.weight.fill_(1 / math.sqrt(2))
        for projection in (
            transformer.seq_attn.query_proj,
            transformer.seq_attn.key_proj,
            transformer.seq_attn.value_proj,
            transformer.seq_attn.out_proj,
        ):
            projection.weight.copy_(torch.eye(2, device=device))
        transformer.ff1.weight.zero_()
    inputs = torch.eye(2, device=device)[None, None]
    patch_mask = torch.zeros(1, 1, 2, dtype=torch.bool, device=device)

    output = transformer(inputs, patch_mask)[0]

    # Full-layer values use sigmoid(1) or sigmoid(1 / sqrt(2)).
    torch.testing.assert_close(output[0, 0, 1, 1], inputs.new_tensor(expected))


@withCUDA
def test_mixing_transformer_residuals(device: torch.device) -> None:
    transformer = MixingTransformer(
        _transformer_config(),
        use_variate_attention=False,
        device=device,
    )
    with torch.no_grad():
        transformer.seq_attn.out_proj.weight.zero_()
        transformer.ff1.weight.zero_()
    inputs = torch.arange(
        2 * 3 * 4 * 8,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 8)
    patch_mask = torch.zeros(2, 3, 4, device=device, dtype=torch.bool)

    output = transformer(inputs, patch_mask)[0]

    torch.testing.assert_close(output, inputs)


@withCUDA
def test_mixing_transformer_routes_temporal_masks(
    device: torch.device,
) -> None:
    transformer = MixingTransformer(
        _transformer_config(),
        use_variate_attention=False,
        device=device,
    )
    inputs = torch.arange(48, device=device, dtype=torch.float32).reshape(
        1, 2, 3, 8
    )
    patch_mask = torch.tensor(
        [[[False, True, False], [False, False, True]]], device=device
    )

    output, attention_mask = transformer(inputs, patch_mask)
    torch.testing.assert_close(
        attention_mask, make_attn_mask(patch_mask.reshape(2, 3))
    )

    masked_input = inputs.clone()
    masked_input[0, 0, 1, 0] += 100
    masked_output = transformer(masked_input, patch_mask)[0]
    torch.testing.assert_close(masked_output[0, 0, 2], output[0, 0, 2])

    future_input = inputs.clone()
    future_input[0, 0, 2, 0] += 100
    future_output = transformer(future_input, patch_mask)[0]
    torch.testing.assert_close(future_output[0, 0, 0], output[0, 0, 0])


@withCUDA
def test_mixing_transformer_uses_variate_attention_and_rope(
    device: torch.device,
) -> None:
    rope_config = replace(_transformer_config(), use_rope_var=True)
    with_rope = MixingTransformer(rope_config, device=device)
    without_rope = MixingTransformer(_transformer_config(), device=device)
    without_rope.load_state_dict(with_rope.state_dict())

    with torch.no_grad():
        for transformer in (with_rope, without_rope):
            transformer.seq_attn.out_proj.weight.zero_()
            transformer.ff1.weight.zero_()
            transformer.var_attn.query_proj.weight.copy_(
                torch.eye(8, device=device)
            )
            transformer.var_attn.key_proj.weight.copy_(
                torch.eye(8, device=device)
            )
            transformer.var_attn.value_proj.weight.copy_(
                torch.eye(8, device=device)
            )
            transformer.var_attn.out_proj.weight.copy_(
                torch.eye(8, device=device)
            )

    inputs = torch.arange(
        3 * 2 * 8,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 2, 8)
    patch_mask = torch.zeros(1, 3, 2, device=device, dtype=torch.bool)

    rope_output = with_rope(inputs, patch_mask)[0]
    no_rope_output = without_rope(inputs, patch_mask)[0]

    assert not torch.allclose(rope_output, inputs)
    assert not torch.allclose(rope_output, no_rope_output)


@withCUDA
def test_mixing_transformer_variate_attention_isolates_patches_and_masks(
    device: torch.device,
) -> None:
    transformer = MixingTransformer(_transformer_config(), device=device)
    with torch.no_grad():
        transformer.seq_attn.out_proj.weight.zero_()
        transformer.ff1.weight.zero_()
        for projection in (
            transformer.var_attn.query_proj,
            transformer.var_attn.key_proj,
            transformer.var_attn.value_proj,
            transformer.var_attn.out_proj,
        ):
            projection.weight.copy_(torch.eye(8, device=device))

    inputs = torch.zeros(1, 3, 2, 8, device=device)
    inputs[0, 0, 0, 0] = 1
    inputs[0, 2, 0, 1] = 1
    inputs[0, 0, 1, 2] = 1
    inputs[0, 2, 1, 3] = 1
    patch_mask = torch.tensor(
        [[[False, False], [False, True], [True, False]]], device=device
    )
    output = transformer(inputs, patch_mask)[0]

    changed = inputs.clone()
    changed[0, 0, 0, 0] = 0
    changed[0, 0, 0, 1] = 1
    changed_output = transformer(changed, patch_mask)[0]
    assert not torch.allclose(changed_output[0, 1, 0], output[0, 1, 0])

    other_patch = inputs.clone()
    other_patch[0, 0, 1, 2] = 0
    other_patch[0, 0, 1, 5] = 1
    other_patch_output = transformer(other_patch, patch_mask)[0]
    torch.testing.assert_close(other_patch_output[0, 1, 0], output[0, 1, 0])

    masked = inputs.clone()
    masked[0, 2, 0, 1] = 0
    masked[0, 2, 0, 4] = 1
    masked_output = transformer(masked, patch_mask)[0]
    torch.testing.assert_close(masked_output[0, 1, 0], output[0, 1, 0])


@withCUDA
def test_multi_head_attention_manual_fully_masked_uses_finite_bias(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_rotary_position_embeddings=False,
        qk_norm="none",
        use_sdpa=False,
        device=device,
    )
    with torch.no_grad():
        for projection in (
            attention.query_proj,
            attention.key_proj,
            attention.value_proj,
            attention.out_proj,
        ):
            projection.weight.copy_(torch.eye(8, device=device))
    inputs = torch.ones(1, 3, 8, device=device)
    patch_mask = torch.ones(1, 3, device=device, dtype=torch.bool)

    output = attention(inputs, patch_mask=patch_mask)[0]

    torch.testing.assert_close(output, inputs)


@withCUDA
@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("rescale_logits", [False, True])
def test_multi_head_attention_matches_google_reference(
    device: torch.device,
    use_sdpa: bool,
    rescale_logits: bool,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=4,
        use_sdpa=use_sdpa,
        rescale_logits=rescale_logits,
        device=device,
    )
    with torch.no_grad():
        for projection in (
            attention.query_proj,
            attention.key_proj,
            attention.value_proj,
            attention.out_proj,
        ):
            projection.weight.copy_(torch.eye(4, device=device))
    inputs = (
        torch.arange(1, 13, device=device, dtype=torch.float32).reshape(
            1, 3, 4
        )
        / 10
    )
    patch_mask = torch.tensor([[False, True, False]], device=device)

    output = attention(inputs, patch_mask=patch_mask)[0]

    # Generated with google-research/timesfm at e31dadd84cb26bd5.
    expected_last = {
        False: [0.828331590, 0.928331673, 1.047185063, 1.147185087],
        True: [0.769981503, 0.869981468, 0.993489444, 1.093489409],
    }[rescale_logits]
    expected = inputs.clone()
    expected[:, 1] = inputs[:, 0]
    expected[:, 2] = output.new_tensor(expected_last)
    torch.testing.assert_close(output, expected)


@withCUDA
def test_multi_head_attention_loads_from_meta(device: torch.device) -> None:
    expected = MultiHeadAttention(num_heads=2, in_features=8, device=device)
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        device="meta",
    )

    attention.load_state_dict(expected.state_dict(), assign=True)
    inputs = torch.arange(
        24,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 8)
    output = attention(inputs)[0]

    assert all(buffer.device == device for buffer in attention.buffers())
    torch.testing.assert_close(output, expected(inputs)[0])


@withCUDA
def test_multi_head_attention_uses_output_projection(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        device=device,
    )
    with torch.no_grad():
        attention.out_proj.weight.zero_()
    inputs = torch.arange(
        24,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 8)

    output = attention(inputs)[0]

    torch.testing.assert_close(output, torch.zeros_like(output))
