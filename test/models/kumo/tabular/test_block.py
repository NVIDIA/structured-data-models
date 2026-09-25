# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import cast

import pytest
import torch
from torch.nn import Linear, Sequential

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.nn import LogScale, RotaryEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_transformer_block(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("channels", [128, 256])
def test_transformer_block_rope(
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> None:
    block = KumoTabularTransformerBlock(
        channels=channels,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        rope=RotaryEmbedding(
            channels=channels // 4,
            layout="split_half",
            requires_grad=False,
            device=device,
        ),
        device=device,
    )
    with torch.no_grad():
        block.attn.out_lin.weight.normal_(std=0.02)
        cast(Linear, cast(Sequential, block.mlp)[-1]).weight.normal_(std=0.02)

    reference = copy.deepcopy(block)
    reference.attn.query_transform = Sequential(
        *cast(Sequential, reference.attn.query_transform)
    )
    reference.attn.key_transform = Sequential(
        *cast(Sequential, reference.attn.key_transform)
    )
    block.load_state_dict(reference.state_dict(), strict=True)
    query = torch.randn(2, 17, channels, device=device, dtype=dtype)
    key_value = torch.randn(2, 13, channels, device=device, dtype=dtype)

    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype):
        actual = block(query=query, key_value=key_value)
        expected = reference(query=query, key_value=key_value)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
