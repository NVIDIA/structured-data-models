# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import pytest
import torch
from torch.nn import Sequential

from sdm.models.timesfm3.transformer import (
    MixingTransformer,
    StackedMixingTransformer,
    TimesFM3Attention,
    make_attn_mask,
)
from sdm.nn import RotaryEmbedding
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


def _set_identity_projections(attention: TimesFM3Attention) -> None:
    channels = attention.qkv_lin.in_features
    identity = torch.eye(
        channels,
        device=attention.qkv_lin.weight.device,
        dtype=attention.qkv_lin.weight.dtype,
    )
    with torch.no_grad():
        attention.qkv_lin.weight.copy_(identity.repeat(3, 1))
        attention.out_lin.weight.copy_(identity)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_rope_matches_google_schedule(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    attention = TimesFM3Attention(
        model_dims=4, num_heads=1, device=device, dtype=dtype
    )
    query_transform = attention.query_transform
    assert isinstance(query_transform, Sequential)
    rope = query_transform[0]
    assert isinstance(rope, RotaryEmbedding)
    inputs = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 0.0]]]],
        device=device,
        dtype=dtype,
    )

    output = rope(inputs)

    expected = torch.tensor(
        [math.cos(1.0), math.cos(0.01), math.sin(1.0), math.sin(0.01)],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(output[0, 0], inputs[0, 0])
    torch.testing.assert_close(output[0, 1, 0], expected)
    torch.testing.assert_close(
        output.float().square().sum(-1),
        inputs.float().square().sum(-1),
        atol=2e-2 if dtype == torch.bfloat16 else 1e-6,
        rtol=0,
    )


@withCUDA
def test_attention_bfloat16(device: torch.device) -> None:
    attention = TimesFM3Attention(
        model_dims=8,
        num_heads=2,
        device=device,
        dtype=torch.bfloat16,
    )
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.bfloat16)

    output = attention(inputs)

    assert output.dtype == torch.bfloat16
    assert output.isfinite().all()


@withCUDA
def test_attention_sdpa_fully_masked(device: torch.device) -> None:
    attention = TimesFM3Attention(
        model_dims=8,
        num_heads=2,
        device=device,
        dtype=torch.float16,
    )
    _set_identity_projections(attention)
    inputs = torch.ones(2, 3, 8, device=device, dtype=torch.float16)
    patch_mask = torch.tensor([[True, False, False]] * 2, device=device)

    output = attention(inputs, attn_mask=make_attn_mask(patch_mask).squeeze(1))

    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    assert output[:, 1:].abs().sum() > 0


@withCUDA
@pytest.mark.parametrize("use_memory_efficient_attention", [True, False])
def test_attention_matches_google_reference(
    device: torch.device,
    use_memory_efficient_attention: bool,
) -> None:
    attention = TimesFM3Attention(
        model_dims=4,
        num_heads=2,
        use_memory_efficient_attention=use_memory_efficient_attention,
        device=device,
    )
    _set_identity_projections(attention)
    inputs = (
        torch.arange(1, 13, device=device, dtype=torch.float32).reshape(
            1, 3, 4
        )
        / 10
    )
    patch_mask = torch.tensor([[False, True, False]], device=device)

    output = attention(inputs, attn_mask=make_attn_mask(patch_mask).squeeze(1))

    # Generated with google-research/timesfm at e31dadd84cb26bd5.
    expected_last = {
        True: [0.828331590, 0.928331673, 1.047185063, 1.147185087],
        False: [0.769981503, 0.869981468, 0.993489444, 1.093489409],
    }[use_memory_efficient_attention]
    expected = inputs.clone()
    expected[:, 1] = inputs[:, 0]
    expected[:, 2] = output.new_tensor(expected_last)
    torch.testing.assert_close(output, expected)


@withCUDA
def test_attention_output_projection_and_autograd(
    device: torch.device,
) -> None:
    attention = TimesFM3Attention(model_dims=8, num_heads=2, device=device)
    _set_identity_projections(attention)
    inputs = (
        torch.arange(24, device=device, dtype=torch.float32)
        .reshape(1, 3, 8)
        .requires_grad_()
    )

    output = attention(inputs)
    output.square().sum().backward()

    assert inputs.grad is not None
    assert inputs.grad.isfinite().all()
    assert attention.qkv_lin.weight.grad is not None
    with torch.no_grad():
        attention.out_lin.weight.zero_()
    torch.testing.assert_close(attention(inputs), torch.zeros_like(output))


def _mixing(
    device: torch.device | str,
    **overrides: Any,
) -> MixingTransformer:
    options: dict[str, Any] = {
        "model_dims": 8,
        "hidden_dims": 12,
        "num_heads": 2,
        "qk_norm": "rms",
        "use_bias": False,
        "use_rope_seq": True,
        "use_rope_var": False,
    }
    options.update(overrides)
    return MixingTransformer(**options, device=device)


@withCUDA
@pytest.mark.parametrize(
    ("memory_efficient", "expected"),
    [(True, 2.3272503), (False, 2.2684078)],
)
def test_mixing_transformer_preserves_google_attention_scale(
    device: torch.device,
    memory_efficient: bool,
    expected: float,
) -> None:
    transformer = _mixing(
        device,
        model_dims=2,
        num_heads=1,
        qk_norm="none",
        use_rope_seq=False,
        use_variate_attention=False,
        use_memory_efficient_attention=memory_efficient,
    )
    with torch.no_grad():
        transformer.pre_seq_attn_ln.weight.fill_(1 / math.sqrt(2))
        _set_identity_projections(transformer.seq_attn)
        transformer.ff1.weight.zero_()
    inputs = torch.eye(2, device=device)[None, None]
    patch_mask = torch.zeros(1, 1, 2, dtype=torch.bool, device=device)

    output, _ = transformer(inputs, patch_mask)

    torch.testing.assert_close(output[0, 0, 1, 1], inputs.new_tensor(expected))


@withCUDA
def test_mixing_transformer_residuals(device: torch.device) -> None:
    transformer = _mixing(device, use_variate_attention=False)
    with torch.no_grad():
        transformer.seq_attn.out_lin.weight.zero_()
        transformer.ff1.weight.zero_()
    inputs = torch.arange(
        2 * 3 * 4 * 8, device=device, dtype=torch.float32
    ).reshape(2, 3, 4, 8)
    patch_mask = torch.zeros(2, 3, 4, device=device, dtype=torch.bool)

    output, _ = transformer(inputs, patch_mask)

    torch.testing.assert_close(output, inputs)


@withCUDA
def test_mixing_transformer_uses_relu(device: torch.device) -> None:
    transformer = _mixing(
        device,
        model_dims=2,
        hidden_dims=2,
        num_heads=1,
        qk_norm="none",
        use_rope_seq=False,
        use_variate_attention=False,
    )
    with torch.no_grad():
        transformer.seq_attn.out_lin.weight.zero_()
        transformer.ff0.weight.copy_(torch.eye(2, device=device))
        transformer.ff1.weight.copy_(torch.eye(2, device=device))
    inputs = torch.tensor([[[[-3.0, 4.0], [-3.0, 4.0]]]], device=device)
    patch_mask = torch.zeros(1, 1, 2, dtype=torch.bool, device=device)

    output, _ = transformer(inputs, patch_mask)

    torch.testing.assert_close(output[..., 0], inputs[..., 0])
    assert (output[..., 1] > inputs[..., 1]).all()


@withCUDA
@pytest.mark.parametrize("causal", [True, False])
def test_mixing_transformer_routes_temporal_masks(
    device: torch.device,
    causal: bool,
) -> None:
    transformer = _mixing(
        device, use_variate_attention=False, causal_attention=causal
    )
    with torch.no_grad():
        _set_identity_projections(transformer.seq_attn)
        transformer.ff1.weight.zero_()
    inputs = torch.arange(48, device=device, dtype=torch.float32).reshape(
        1, 2, 3, 8
    )
    patch_mask = torch.tensor(
        [[[False, True, False], [False, False, True]]], device=device
    )

    output, attention_mask = transformer(inputs, patch_mask)
    torch.testing.assert_close(
        attention_mask,
        make_attn_mask(patch_mask.reshape(2, 3), causal=causal),
    )

    masked_input = inputs.clone()
    masked_input[0, 0, 1, 0] += 100
    masked_output = transformer(masked_input, patch_mask)[0]
    torch.testing.assert_close(masked_output[0, 0, 2], output[0, 0, 2])

    future_input = inputs.clone()
    future_input[0, 0, 2, 0] += 100
    future_output = transformer(future_input, patch_mask)[0]
    if causal:
        torch.testing.assert_close(future_output[0, 0, 0], output[0, 0, 0])
    else:
        assert not torch.allclose(future_output[0, 0, 0], output[0, 0, 0])

    other_variate = inputs.clone()
    other_variate[0, 1, 0, 0] += 100
    other_output = transformer(other_variate, patch_mask)[0]
    torch.testing.assert_close(other_output[0, 0], output[0, 0])


@withCUDA
def test_mixing_transformer_variate_attention_isolates_patches_and_masks(
    device: torch.device,
) -> None:
    transformer = _mixing(device)
    with torch.no_grad():
        transformer.seq_attn.out_lin.weight.zero_()
        transformer.ff1.weight.zero_()
        _set_identity_projections(transformer.var_attn)

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
def test_mixing_transformer_variate_rope_is_configurable(
    device: torch.device,
) -> None:
    with_rope = _mixing(device, use_rope_var=True)
    without_rope = _mixing(device)
    without_rope.load_state_dict(with_rope.state_dict(), strict=False)
    with torch.no_grad():
        for transformer in (with_rope, without_rope):
            transformer.seq_attn.out_lin.weight.zero_()
            transformer.ff1.weight.zero_()
            _set_identity_projections(transformer.var_attn)

    inputs = torch.arange(
        3 * 2 * 8, device=device, dtype=torch.float32
    ).reshape(1, 3, 2, 8)
    patch_mask = torch.zeros(1, 3, 2, device=device, dtype=torch.bool)

    rope_output = with_rope(inputs, patch_mask)[0]
    no_rope_output = without_rope(inputs, patch_mask)[0]

    assert not torch.allclose(rope_output, inputs)
    assert not torch.allclose(rope_output, no_rope_output)


@withCUDA
def test_mixing_transformer_bfloat16_meta_loading(
    device: torch.device,
) -> None:
    expected = _mixing(device, use_rope_var=True, dtype=torch.bfloat16)
    transformer = _mixing("meta", use_rope_var=True, dtype=torch.bfloat16)
    assert all(
        parameter.device.type == "meta"
        for parameter in transformer.parameters()
    )
    assert all(
        buffer.device.type == "meta" for buffer in transformer.buffers()
    )

    transformer.load_state_dict(
        expected.state_dict(), strict=True, assign=True
    )
    inputs = torch.arange(48, device=device, dtype=torch.bfloat16).reshape(
        1, 3, 2, 8
    )
    patch_mask = torch.zeros(1, 3, 2, dtype=torch.bool, device=device)

    output, mask = transformer(inputs, patch_mask)

    assert output.dtype == torch.bfloat16
    assert output.isfinite().all()
    assert mask.device == device
    torch.testing.assert_close(output, expected(inputs, patch_mask)[0])


def _stack(
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> StackedMixingTransformer:
    return StackedMixingTransformer(
        num_layers=2,
        model_dims=8,
        hidden_dims=12,
        num_heads=2,
        qk_norm="rms",
        use_bias=False,
        use_rope_seq=True,
        use_rope_var=False,
        use_variate_attention=False,
        device=device,
        dtype=dtype,
    )


@withCUDA
def test_stacked_transformer_chains_layers_and_masks(
    device: torch.device,
) -> None:
    transformer = _stack(device)
    with torch.no_grad():
        for layer in transformer.layers:
            assert isinstance(layer, MixingTransformer)
            _set_identity_projections(layer.seq_attn)
            layer.ff1.weight.fill_(0.02)
    inputs = torch.arange(192, device=device, dtype=torch.float32).reshape(
        2, 3, 4, 8
    )
    patch_mask = torch.zeros(2, 3, 4, dtype=torch.bool, device=device)
    patch_mask[0, 0, 1] = True
    patch_mask[1, 2, 2] = True

    output, masks = transformer(inputs, patch_mask)
    first_output, _ = transformer.layers[0](inputs, patch_mask)
    expected_output, _ = transformer.layers[1](first_output, patch_mask)
    expected_mask = make_attn_mask(patch_mask.reshape(2 * 3, 4))

    torch.testing.assert_close(output, expected_output)
    assert not torch.allclose(output, inputs)
    assert len(masks) == 2
    for mask in masks:
        torch.testing.assert_close(mask, expected_mask)


@withCUDA
def test_stacked_transformer_strict_meta_bfloat16_load(
    device: torch.device,
) -> None:
    expected = _stack(device, dtype=torch.bfloat16)
    transformer = _stack("meta", dtype=torch.bfloat16)
    assert all(
        parameter.device.type == "meta"
        for parameter in transformer.parameters()
    )
    assert all(
        buffer.device.type == "meta" for buffer in transformer.buffers()
    )

    transformer.load_state_dict(
        expected.state_dict(), strict=True, assign=True
    )
    inputs = torch.arange(64, device=device, dtype=torch.bfloat16).reshape(
        1, 2, 4, 8
    )
    patch_mask = torch.zeros(1, 2, 4, dtype=torch.bool, device=device)

    output, masks = transformer(inputs, patch_mask)

    assert output.dtype == torch.bfloat16
    assert output.isfinite().all()
    assert len(masks) == 2
    assert all(buffer.device == device for buffer in transformer.buffers())
    torch.testing.assert_close(output, expected(inputs, patch_mask)[0])
