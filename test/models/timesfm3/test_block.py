# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from sdm.models.timesfm3.block import MixingTransformerBlock, ResidualBlock
from sdm.testing import withCUDA


@withCUDA
def test_residual_block(device: torch.device) -> None:
    block = ResidualBlock(
        in_channels=3,
        out_channels=2,
        device=device,
    )

    out = block(torch.randn(1, 3, device=device))
    assert out.size() == (1, 2)


@withCUDA
@pytest.mark.parametrize("masked", [False, True])
def test_mixing_transformer_block(
    device: torch.device,
    masked: bool,
) -> None:
    block = MixingTransformerBlock(
        channels=8,
        num_heads=2,
        device=device,
    )

    x = torch.randn(2, 3, 4, 8, device=device)
    patch_mask = None
    if masked:
        patch_mask = torch.randn(2, 3, 4, device=device) >= 0

    out = block(x, patch_mask)
    assert out.size() == x.size()
