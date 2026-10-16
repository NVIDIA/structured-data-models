# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.nn import SinusoidalEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16]
)
@pytest.mark.parametrize("channels", [3, 4])
def test_sinusoidal_embedding(
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> None:
    module = SinusoidalEmbedding()
    x = torch.ones(2, 3, channels, device=device, dtype=dtype)
    x.requires_grad_()
    positions = x.new_tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)],
            [math.sin(2), math.cos(2), math.sin(0.02), math.cos(0.02)],
        ]
    )
    if channels == 3:
        frequency = 10000 ** (-2 / 3)
        positions[:, 2] = x.new_tensor(
            [0, math.sin(frequency), math.sin(2 * frequency)]
        )
    expected = x + positions[:, :channels]
    out = module(x)
    assert out.device == device
    assert out.dtype == dtype
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@withCUDA
def test_sinusoidal_embedding_theta(device: torch.device) -> None:
    module = SinusoidalEmbedding(theta=1)
    x = torch.zeros(2, 4, device=device)
    expected = x.new_tensor([[0, 1, 0, 1], [math.sin(1), math.cos(1)] * 2])
    torch.testing.assert_close(module(x), expected)


@withCUDA
def test_sinusoidal_embedding_variable_length(device: torch.device) -> None:
    module = SinusoidalEmbedding()
    x = torch.randn(2, 129, 8, device=device)
    out = module(x)
    assert out.shape == x.shape
    for length in [1, 64, 128]:
        torch.testing.assert_close(
            module(x[..., :length, :]), out[..., :length, :]
        )
