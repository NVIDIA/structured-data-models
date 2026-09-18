# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import pytest
import torch

from sdm.models.timesfm3.configs import (
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.transformer import (
    MixingTransformer,
    MultiHeadAttention,
    RotaryPositionalEmbedding,
    StackedMixingTransformer,
    make_attn_mask,
    make_segment_mask,
)
from sdm.models.timesfm3.util import DecodeCache
from sdm.testing import withCUDA


@withCUDA
def test_make_attn_mask(device: torch.device) -> None:
    num_masked = torch.tensor([1, 2], device=device)
    query_offset = torch.tensor([0, 2], device=device)

    mask = make_attn_mask(
        query_length=2,
        num_all_masked_kv=num_masked,
        query_index_offset=query_offset,
        kv_length=4,
    )

    expected = torch.tensor(
        [
            [[[False, False, False, False], [False, True, False, False]]],
            [[[False, False, True, False], [False, False, True, True]]],
        ],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_attn_mask_noncausal(device: torch.device) -> None:
    num_masked = torch.tensor([2], device=device)

    mask = make_attn_mask(
        query_length=3,
        num_all_masked_kv=num_masked,
        kv_length=4,
        causal=False,
    )

    expected = torch.tensor(
        [[[[False, False, True, True]]]],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_segment_mask(device: torch.device) -> None:
    segment_ids = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 2]],
        device=device,
    )

    mask = make_segment_mask(segment_ids)

    expected = segment_ids[:, :, None] == segment_ids[:, None, :]
    torch.testing.assert_close(mask, expected.unsqueeze(1))


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
    positions = torch.tensor([[0, 1, 2], [2, 3, 4]], device=device)

    output = rope(inputs, position=positions)

    torch.testing.assert_close(
        output.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
    )
    torch.testing.assert_close(output[0, 0], inputs[0, 0])
    assert output.shape == inputs.shape
    assert output.device == device


@withCUDA
def test_rotary_positional_embedding_uses_sequential_positions(
    device: torch.device,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    inputs = torch.arange(24, device=device, dtype=torch.float32).reshape(
        2, 3, 4
    )
    positions = torch.arange(3, device=device).unsqueeze(0)

    torch.testing.assert_close(rope(inputs), rope(inputs, position=positions))


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
def test_make_attn_mask_uses_query_length_by_default(
    device: torch.device,
) -> None:
    num_masked = torch.tensor([1], device=device)

    mask = make_attn_mask(query_length=3, num_all_masked_kv=num_masked)

    expected = torch.tensor(
        [
            [
                [
                    [False, False, False],
                    [False, True, False],
                    [False, True, True],
                ]
            ]
        ],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_rotary_positional_embedding_rotation_direction(
    device: torch.device,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=2, device=device)
    inputs = torch.tensor([[[1.0, 0.0]]], device=device)
    positions = torch.tensor([[1]], device=device)

    output = rope(inputs, position=positions)

    expected = torch.tensor(
        [[[math.cos(1.0), math.sin(1.0)]]],
        device=device,
    )
    torch.testing.assert_close(output, expected)


def _transformer_config(use_sdpa: bool = True) -> TransformerConfig:
    return TransformerConfig(
        model_dims=8,
        hidden_dims=12,
        num_heads=2,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        ff_activation="relu",
        deterministic=True,
        use_sdpa=use_sdpa,
    )


@withCUDA
def test_multi_head_attention_sdpa_matches_manual(
    device: torch.device,
) -> None:
    sdpa = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=True,
        device=device,
    )
    manual = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=False,
        device=device,
    )
    manual.load_state_dict(sdpa.state_dict())
    inputs = (
        torch.arange(
            64,
            device=device,
            dtype=torch.float32,
        ).reshape(2, 4, 8)
        / 64
    )
    patch_mask = torch.tensor(
        [[True, False, False, False], [False, False, True, False]],
        device=device,
    )

    sdpa_output, _, sdpa_mask = sdpa(inputs, patch_mask=patch_mask)
    manual_output, _, manual_mask = manual(inputs, patch_mask=patch_mask)

    torch.testing.assert_close(sdpa_mask, manual_mask)
    torch.testing.assert_close(sdpa_output, manual_output)


@withCUDA
def test_stacked_transformer_cached_matches_full_sequence(
    device: torch.device,
) -> None:
    config = StackedTransformersConfig(
        num_layers=2,
        transformer=_transformer_config(),
    )
    transformer = StackedMixingTransformer(config, device=device)
    inputs = (
        torch.arange(
            2 * 3 * 4 * 8,
            device=device,
            dtype=torch.float32,
        ).reshape(2, 3, 4, 8)
        / 100
    )
    patch_mask = torch.zeros(2, 3, 4, dtype=torch.bool, device=device)
    patch_mask[0, :, 0] = True
    patch_mask[1, :, 2] = True
    segment_ids = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 0, 1]],
        device=device,
    )

    full_output, full_cache, full_masks = transformer(
        inputs,
        patch_mask,
        segment_ids=segment_ids,
    )

    caches = DecodeCache.init_decode_cache(
        num_layers=2,
        batch_size=2,
        num_variates=3,
        num_total_input_patches=4,
        num_heads=2,
        head_dim=4,
        device=device,
    )
    cached_outputs = []
    cached_masks = []
    with torch.inference_mode():
        for index in range(4):
            output, caches, masks = transformer(
                inputs[:, :, index : index + 1],
                patch_mask[:, :, index : index + 1],
                segment_ids=segment_ids[:, index : index + 1],
                decode_cache=caches,
            )
            cached_outputs.append(output)
            cached_masks.append(masks)
            assert all(mask.size(-1) == index + 1 for mask in masks)

    assert all(cache is None for cache in full_cache)
    assert len(full_masks) == 2
    assert len(caches) == 2
    assert all(
        cache is not None and cache.next_index.eq(4).all() for cache in caches
    )
    assert all(len(masks) == 2 for masks in cached_masks)
    torch.testing.assert_close(
        torch.cat(cached_outputs, dim=2),
        full_output,
        atol=2e-5,
        rtol=2e-5,
    )


@withCUDA
def test_cached_attention_requires_inference_mode(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        device=device,
    )
    cache = DecodeCache.init_decode_cache(
        num_layers=1,
        batch_size=1,
        num_variates=1,
        num_total_input_patches=2,
        num_heads=2,
        head_dim=4,
        device=device,
    )[0]

    with pytest.raises(RuntimeError, match="requires inference mode"):
        attention(torch.ones(1, 1, 8, device=device), decode_cache=cache)


@withCUDA
def test_stacked_transformer_shape(device: torch.device) -> None:
    transformer = StackedMixingTransformer(
        StackedTransformersConfig(
            num_layers=2,
            transformer=_transformer_config(),
        ),
        device=device,
    )
    inputs = torch.zeros(2, 3, 4, 8, device=device)
    patch_mask = torch.zeros(2, 3, 4, dtype=torch.bool, device=device)

    output, caches, masks = transformer(inputs, patch_mask)

    assert output.shape == inputs.shape
    assert len(transformer.layers) == 2
    assert all(cache is None for cache in caches)
    assert len(masks) == 2
    assert all(mask.shape == (2 * 3, 1, 4, 4) for mask in masks)


@withCUDA
def test_stacked_transformer_loads_from_meta(
    device: torch.device,
) -> None:
    config = StackedTransformersConfig(
        num_layers=2,
        transformer=_transformer_config(),
    )
    expected = StackedMixingTransformer(config, device=device)
    transformer = StackedMixingTransformer(config, device="meta")
    assert all(
        buffer.device.type == "meta" for buffer in transformer.buffers()
    )

    transformer.load_state_dict(expected.state_dict(), assign=True)
    inputs = torch.arange(
        64,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 8, 8)
    patch_mask = torch.zeros(1, 1, 8, dtype=torch.bool, device=device)

    output = transformer(inputs, patch_mask)[0]

    assert all(buffer.device == device for buffer in transformer.buffers())
    torch.testing.assert_close(output, expected(inputs, patch_mask)[0])


@withCUDA
def test_multi_head_attention_bfloat16(device: torch.device) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=True,
        device=device,
        dtype=torch.bfloat16,
    )
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.bfloat16)

    output = attention(inputs)[0]

    assert output.dtype == torch.bfloat16
    assert output.isfinite().all()


@withCUDA
def test_multi_head_attention_manual_fully_masked(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=False,
        device=device,
        dtype=torch.float16,
    )
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.float16)
    patch_mask = torch.ones(2, 3, device=device, dtype=torch.bool)

    output = attention(inputs, patch_mask=patch_mask)[0]

    torch.testing.assert_close(output, torch.zeros_like(output))


@withCUDA
def test_multi_head_attention_uses_output_projection(
    device: torch.device,
) -> None:
    attention = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        use_sdpa=True,
        device=device,
    )
    attention.out_proj.weight.data.zero_()
    inputs = torch.arange(
        24,
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3, 8)

    output = attention(inputs)[0]

    torch.testing.assert_close(output, torch.zeros_like(output))


@withCUDA
def test_mixing_transformer_residuals(device: torch.device) -> None:
    transformer = MixingTransformer(
        _transformer_config(),
        use_variate_attention=False,
        device=device,
    )
    transformer.seq_attn.out_proj.weight.data.zero_()
    transformer.ff1.weight.data.zero_()
    inputs = torch.arange(
        2 * 3 * 4 * 8,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 3, 4, 8)
    patch_mask = torch.zeros(2, 3, 4, device=device, dtype=torch.bool)

    output = transformer(inputs, patch_mask)[0]

    torch.testing.assert_close(output, inputs)


@withCUDA
def test_mixing_transformer_uses_variate_attention_and_rope(
    device: torch.device,
) -> None:
    rope_config = replace(_transformer_config(), use_rope_var=True)
    with_rope = MixingTransformer(rope_config, device=device)
    without_rope = MixingTransformer(_transformer_config(), device=device)
    without_rope.load_state_dict(with_rope.state_dict())

    for transformer in (with_rope, without_rope):
        transformer.seq_attn.out_proj.weight.data.zero_()
        transformer.ff1.weight.data.zero_()
        transformer.var_attn.query_proj.weight.data.copy_(
            torch.eye(8, device=device)
        )
        transformer.var_attn.key_proj.weight.data.copy_(
            torch.eye(8, device=device)
        )
        transformer.var_attn.value_proj.weight.data.copy_(
            torch.eye(8, device=device)
        )
        transformer.var_attn.out_proj.weight.data.copy_(
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
def test_multi_head_attention_uses_qk_norm_and_rope(
    device: torch.device,
) -> None:
    baseline = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        qk_norm="none",
        use_rotary_position_embeddings=False,
        use_sdpa=True,
        device=device,
    )
    normalized = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        qk_norm="rms",
        use_rotary_position_embeddings=False,
        use_sdpa=True,
        device=device,
    )
    rotated = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        qk_norm="none",
        use_rotary_position_embeddings=True,
        use_sdpa=True,
        device=device,
    )
    with torch.no_grad():
        for attention in (baseline, normalized, rotated):
            for projection in (
                attention.query_proj,
                attention.key_proj,
                attention.value_proj,
                attention.out_proj,
            ):
                projection.weight.copy_(torch.eye(8, device=device))

    inputs = (
        torch.arange(1, 33, device=device, dtype=torch.float32).reshape(
            1, 4, 8
        )
        / 10
    )

    baseline_output = baseline(inputs)[0]
    normalized_output = normalized(inputs)[0]
    rotated_output = rotated(inputs)[0]

    assert not torch.allclose(normalized_output, baseline_output)
    assert not torch.allclose(rotated_output, baseline_output)
