# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timestral.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
def test_icl_block(device: torch.device) -> None:
    block = ICLBlock(
        channels=16,
        num_layers=4,
        column_every=2,
        num_heads=2,
        hidden_channels=24,
        device=device,
    )

    x = torch.randn(2, 3, 5, 16, device=device)  # [B, C, R, D]
    row_mask = torch.ones(1, 1, 1, 5, 5, device=device, dtype=torch.bool)
    col_mask = torch.ones(1, 1, 1, 3, 3, device=device)
    pos = torch.arange(5, device=device)

    out = block(x, row_mask, col_mask, pos)
    assert out.device == device
    assert out.size() == x.size()
    assert out.isfinite().all()
