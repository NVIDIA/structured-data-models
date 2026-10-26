# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.anomaly.denoiser import DiffusionDenoiser
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize(("features", "length"), [(3, 5), (1, 1)])
def test_diffusion_denoiser(
    input_channels: int,
    features: int,
    length: int,
    device: torch.device,
) -> None:
    model = DiffusionDenoiser(
        channels=8,
        side_channels=6,
        num_steps=10,
        num_layers=2,
        num_heads=2,
        embedding_channels=12,
        input_channels=input_channels,
        device=device,
    )

    x = torch.randn(2, input_channels, features, length, device=device)
    side_info = torch.randn(2, 6, features, length, device=device)
    step = torch.tensor([0, 9], device=device)
    strategy = torch.tensor([1], device=device)

    with torch.no_grad():
        out = model(x, side_info, step, strategy)

    assert out.size() == (2, features, length)
    assert out.dtype == x.dtype
    assert out.device == device
