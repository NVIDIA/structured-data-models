# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.nn import SinusoidalEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_sinusoidal_embedding(device: torch.device) -> None:
    module = SinusoidalEmbedding()
    x = torch.ones(2, 3, 4, device=device)
    expected = x + x.new_tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)],
            [math.sin(2), math.cos(2), math.sin(0.02), math.cos(0.02)],
        ]
    )

    out = module(x)
    assert out.device == device
    torch.testing.assert_close(out, expected)


@withCUDA
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("batched_positions", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sinusoidal_embedding_pos(
    device: torch.device,
    channels: int,
    batched_positions: bool,
    dtype: torch.dtype,
) -> None:
    module = SinusoidalEmbedding(theta=100)
    x = torch.randn(2, 3, 4, channels, device=device, dtype=dtype)
    positions = torch.tensor([-1, 0.5, 2, 0.5], device=device)
    if batched_positions:
        positions = torch.stack([positions, positions + 3]).unsqueeze(1)
    expected_channels = []
    for channel in range(channels):
        angle = positions / (100 ** (2 * (channel // 2) / channels))
        expected_channels.append(
            angle.sin() if channel % 2 == 0 else angle.cos()
        )
    expected = x + torch.stack(expected_channels, dim=-1).to(dtype)

    actual = module(x, pos=positions)
    assert actual.dtype == dtype
    assert actual.device == device
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        module(x),
        module(x, pos=torch.arange(x.size(-2), device=device)),
    )
