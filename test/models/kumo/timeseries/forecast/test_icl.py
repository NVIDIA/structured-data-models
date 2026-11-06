# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length", [9, 140])
@pytest.mark.parametrize(("num_buckets", "max_distance"), [(4, 2), (32, 128)])
def test_icl_block(
    device: torch.device,
    length: int,
    num_buckets: int,
    max_distance: int,
    masked: bool,
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
    patch_mask = None
    if masked:
        patch_mask = torch.zeros(2, length, dtype=torch.bool, device=device)
        patch_mask[:, :3] = True

    with torch.no_grad():
        out = model(x, patch_mask=patch_mask)

    assert out.size() == (2, length, 32)
    assert out.dtype == x.dtype
    assert out.device == device
