# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timesfm3.block import ResidualBlock
from sdm.testing import withCUDA


@withCUDA
def test_residual_block(device: torch.device) -> None:
    block = ResidualBlock(
        in_channels=3,
        out_channels=2,
        bias=False,
        device=device,
    )

    out = block(torch.randn(1, 3, device=device))
    assert out.size() == (1, 2)
