# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.nn import Sequential

from sdm.models.timesfm3.transformer import (
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
