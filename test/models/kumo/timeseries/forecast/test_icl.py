# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("length", [9, 140])
@pytest.mark.parametrize(("num_buckets", "max_distance"), [(4, 2), (32, 128)])
def test_icl_block(
    device: torch.device,
    length: int,
    num_buckets: int,
    max_distance: int,
) -> None:
    model = ICLBlock(
        channels=32,
        hidden_channels=48,
        num_layers=2,
        num_heads=4,
        num_buckets=num_buckets,
        max_distance=max_distance,
        device=device,
    )

    x = torch.randn(2, length, 32, device=device)

    with torch.no_grad():
        out = model(x)

    assert out.size() == (2, length, 32)
    assert out.dtype == x.dtype
    assert out.device == device
