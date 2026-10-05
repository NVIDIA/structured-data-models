# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import Sequential

from sdm.models.timesfm3.transformer import TimesFM3TransformerBlock
from sdm.nn import Attention, SoftplusScale
from sdm.testing import withCUDA


def _set_identity_projections(attention: Attention) -> None:
    channels = attention.qkv_lin.in_features
    identity = torch.eye(
        channels,
        device=attention.qkv_lin.weight.device,
        dtype=attention.qkv_lin.weight.dtype,
    )
    with torch.no_grad():
        attention.qkv_lin.weight.copy_(identity.repeat(3, 1))
        attention.out_lin.weight.copy_(identity)


def _set_scale_weight(attention: Attention, value: float) -> None:
    transform = attention.query_transform
    assert isinstance(transform, Sequential)
    scale = transform[-1]
    assert isinstance(scale, SoftplusScale)
    with torch.no_grad():
        scale.weight.fill_(value)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block_matches_released_reference(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    block = TimesFM3TransformerBlock(4, 2, device=device, dtype=dtype)
    with torch.no_grad():
        identity = torch.eye(4, device=device, dtype=dtype)
        for attention in (block.seq_attn, block.var_attn):
            _set_identity_projections(attention)
            _set_scale_weight(attention, 1)
        block.ff0.weight.copy_(identity)
        block.ff1.weight.copy_(identity * 0.5)

    x = (
        torch.arange(1, 25, device=device, dtype=dtype).reshape(1, 2, 3, 4)
        / 10
    )
    patch_mask = torch.tensor(
        [[[False, True, False], [False, False, True]]], device=device
    )
    output, temporal_mask = block(x, patch_mask)

    # Reference: PR8 MixingTransformer with the released attention settings.
    expected = x.new_tensor(
        [
            [
                [1.5656608, 2.5713470, 3.5739877, 4.5655751],
                [2.4593966, 3.1772628, 3.8951263, 4.6129866],
                [3.4472342, 3.8380885, 4.2283440, 4.6161804],
            ],
            [
                [3.6777005, 4.1635895, 4.6521811, 5.1505814],
                [4.4457173, 4.7114272, 4.9771290, 5.2428198],
                [4.7759833, 5.0856724, 5.3951769, 5.7039437],
            ],
        ]
    ).unsqueeze(0)
    torch.testing.assert_close(
        output,
        expected,
        rtol=2e-2 if dtype == torch.bfloat16 else 1e-4,
        atol=2e-3 if dtype == torch.bfloat16 else 1e-5,
    )
    assert output.dtype == dtype
    assert output.isfinite().all()
    torch.testing.assert_close(
        temporal_mask[:, 0],
        torch.tensor(
            [
                [
                    [True, False, False],
                    [True, False, False],
                    [True, False, True],
                ],
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, False],
                ],
            ],
            device=device,
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    ("scale_weight", "expected_last"),
    [
        (0.0, [0.828331590, 0.928331673, 1.047185063, 1.147185087]),
        (1.0, [0.890230954, 0.990230918, 1.094750404, 1.194750428]),
    ],
)
def test_temporal_attention_matches_google_reference(
    device: torch.device,
    scale_weight: float,
    expected_last: list[float],
) -> None:
    block = TimesFM3TransformerBlock(4, 2, device=device)
    attention = block.seq_attn
    _set_identity_projections(attention)
    _set_scale_weight(attention, scale_weight)
    x = (
        torch.arange(1, 13, device=device, dtype=torch.float32).reshape(
            1, 3, 4
        )
        / 10
    )
    patch_mask = torch.tensor([[False, True, False]], device=device)
    causal = torch.ones(3, 3, device=device, dtype=torch.bool).tril()

    output = attention(x, attn_mask=causal[None] & ~patch_mask[:, None, :])

    # Zero-weight output from google-research/timesfm at e31dadd84cb26bd5.
    expected = x.clone()
    expected[:, 1] = x[:, 0]
    expected[:, 2] = x.new_tensor(expected_last)
    torch.testing.assert_close(output, expected)


@withCUDA
def test_temporal_attention_is_causal_and_ignores_masked_patches(
    device: torch.device,
) -> None:
    block = TimesFM3TransformerBlock(8, 2, device=device)
    _set_identity_projections(block.seq_attn)
    with torch.no_grad():
        block.var_attn.out_lin.weight.zero_()
        block.ff1.weight.zero_()
    x = torch.arange(48, device=device, dtype=torch.float32).reshape(
        1, 2, 3, 8
    )
    patch_mask = torch.tensor(
        [[[False, True, False], [False, False, True]]], device=device
    )
    output, temporal_mask = block(x, patch_mask)

    assert temporal_mask.shape == (2, 1, 3, 3)
    assert not temporal_mask[0, 0, 2, 1]
    assert not temporal_mask[0, 0, 0, 2]

    changed = x.clone()
    changed[0, 0, 1, 0] += 100
    torch.testing.assert_close(
        block(changed, patch_mask)[0][0, 0, 2], output[0, 0, 2]
    )
    changed = x.clone()
    changed[0, 0, 2, 0] += 100
    torch.testing.assert_close(
        block(changed, patch_mask)[0][0, 0, 0], output[0, 0, 0]
    )


@withCUDA
def test_variate_attention_uses_only_same_patch_and_unmasked_keys(
    device: torch.device,
) -> None:
    block = TimesFM3TransformerBlock(8, 2, device=device)
    _set_identity_projections(block.var_attn)
    with torch.no_grad():
        block.seq_attn.out_lin.weight.zero_()
        block.ff1.weight.zero_()

    x = torch.zeros(1, 3, 2, 8, device=device)
    x[0, 0, 0, 0] = 1
    x[0, 2, 0, 1] = 1
    x[0, 0, 1, 2] = 1
    patch_mask = torch.tensor(
        [[[False, False], [False, True], [True, False]]], device=device
    )
    output = block(x, patch_mask)[0]

    changed = x.clone()
    changed[0, 0, 0, 0] = 0
    changed[0, 0, 0, 1] = 1
    assert not torch.allclose(
        block(changed, patch_mask)[0][0, 1, 0], output[0, 1, 0]
    )

    changed = x.clone()
    changed[0, 0, 1, 2] = 0
    changed[0, 0, 1, 5] = 1
    torch.testing.assert_close(
        block(changed, patch_mask)[0][0, 1, 0], output[0, 1, 0]
    )

    changed = x.clone()
    changed[0, 2, 0, 1] = 0
    changed[0, 2, 0, 4] = 1
    torch.testing.assert_close(
        block(changed, patch_mask)[0][0, 1, 0], output[0, 1, 0]
    )


@withCUDA
def test_block_residuals_and_relu(device: torch.device) -> None:
    block = TimesFM3TransformerBlock(4, 1, device=device)
    with torch.no_grad():
        block.seq_attn.out_lin.weight.zero_()
        block.var_attn.out_lin.weight.zero_()
        block.ff1.weight.zero_()
    x = torch.tensor([[[[-3.0, 4.0, -1.0, 2.0]]]], device=device)
    patch_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    torch.testing.assert_close(block(x, patch_mask)[0], x)

    with torch.no_grad():
        identity = torch.eye(4, device=device)
        block.ff0.weight.copy_(identity)
        block.ff1.weight.copy_(identity)
    output = block(x, patch_mask)[0]
    torch.testing.assert_close(output[..., 0], x[..., 0])
    assert (output[..., 1] > x[..., 1]).all()
