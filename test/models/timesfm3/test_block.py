# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.nn import Linear, Sequential

from sdm.models.timesfm3.block import TimesFM3TransformerBlock
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


def _ff_layer(block: TimesFM3TransformerBlock, name: str) -> Linear:
    layer = block.ff_block.get_submodule(name)
    assert isinstance(layer, Linear)
    return layer


@withCUDA
@pytest.mark.parametrize("scale_weight", [0.0, 1.0])
def test_variate_attention_query_scale(
    device: torch.device,
    scale_weight: float,
) -> None:
    block = TimesFM3TransformerBlock(2, 1, device=device)
    attention = block.var_attn
    _set_identity_projections(attention)
    _set_scale_weight(attention, scale_weight)
    x = torch.eye(2, device=device).unsqueeze(0)

    output = attention(x)

    # RMSNorm makes each matching query-key dot product equal to 2.
    query_scale = math.log1p(math.exp(scale_weight)) / math.log(2)
    same_token_weight = torch.sigmoid(x.new_tensor(2 * query_scale))
    expected = torch.stack((same_token_weight, 1 - same_token_weight))
    torch.testing.assert_close(output[0, 0], expected)
    torch.testing.assert_close(output[0, 1], expected.flip(0))


@withCUDA
def test_temporal_attention_is_causal_and_ignores_masked_patches(
    device: torch.device,
) -> None:
    block = TimesFM3TransformerBlock(8, 2, device=device)
    _set_identity_projections(block.time_attn)
    with torch.no_grad():
        block.var_attn.out_lin.weight.zero_()
        _ff_layer(block, "ff1").weight.zero_()
    x = torch.arange(48, device=device, dtype=torch.float32).reshape(
        1, 2, 3, 8
    )
    patch_mask = torch.tensor(
        [[[False, True, False], [False, False, True]]], device=device
    )
    output = block(x, patch_mask)

    changed = x.clone()
    changed[0, 0, 1, 0] += 100
    torch.testing.assert_close(
        block(changed, patch_mask)[0, 0, 2], output[0, 0, 2]
    )
    changed = x.clone()
    changed[0, 0, 2, 0] += 100
    torch.testing.assert_close(
        block(changed, patch_mask)[0, 0, 0], output[0, 0, 0]
    )


@withCUDA
def test_variate_attention_uses_only_same_patch_and_unmasked_keys(
    device: torch.device,
) -> None:
    block = TimesFM3TransformerBlock(8, 2, device=device)
    _set_identity_projections(block.var_attn)
    with torch.no_grad():
        block.time_attn.out_lin.weight.zero_()
        _ff_layer(block, "ff1").weight.zero_()

    x = torch.zeros(1, 3, 2, 8, device=device)
    x[0, 0, 0, 0] = 1
    x[0, 2, 0, 1] = 1
    x[0, 0, 1, 2] = 1
    patch_mask = torch.tensor(
        [[[False, False], [False, True], [True, False]]], device=device
    )
    output = block(x, patch_mask)

    changed = x.clone()
    changed[0, 0, 0, 0] = 0
    changed[0, 0, 0, 1] = 1
    assert not torch.allclose(
        block(changed, patch_mask)[0, 1, 0], output[0, 1, 0]
    )

    changed = x.clone()
    changed[0, 0, 1, 2] = 0
    changed[0, 0, 1, 5] = 1
    torch.testing.assert_close(
        block(changed, patch_mask)[0, 1, 0], output[0, 1, 0]
    )

    changed = x.clone()
    changed[0, 2, 0, 1] = 0
    changed[0, 2, 0, 4] = 1
    torch.testing.assert_close(
        block(changed, patch_mask)[0, 1, 0], output[0, 1, 0]
    )


@withCUDA
def test_block_residuals_and_relu(device: torch.device) -> None:
    block = TimesFM3TransformerBlock(4, 1, device=device)
    with torch.no_grad():
        block.time_attn.out_lin.weight.zero_()
        block.var_attn.out_lin.weight.zero_()
        _ff_layer(block, "ff1").weight.zero_()
    x = torch.tensor([[[[-3.0, 4.0, -1.0, 2.0]]]], device=device)
    patch_mask = torch.zeros(1, 1, 1, dtype=torch.bool, device=device)
    torch.testing.assert_close(block(x, patch_mask), x)

    with torch.no_grad():
        identity = torch.eye(4, device=device)
        _ff_layer(block, "ff0").weight.copy_(identity)
        _ff_layer(block, "ff1").weight.copy_(identity)
    output = block(x, patch_mask)
    torch.testing.assert_close(output[..., 0], x[..., 0])
    assert (output[..., 1] > x[..., 1]).all()
