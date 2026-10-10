# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from sdm.models.timesfm3.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("masked", [False, True])
def test_icl_block(device: torch.device, masked: bool) -> None:
    block = ICLBlock(
        out_channels=4,
        channels=8,
        num_layers=2,
        num_heads=2,
        device=device,
    )

    x = torch.randn(2, 3, 4, 8, device=device)
    patch_mask = torch.randn(2, 3, 4, device=device) >= 0 if masked else None

    out = block(x, patch_mask)
    assert out.size() == (*x.size()[:-1], 4)

    with torch.inference_mode():
        torch.testing.assert_close(block(x, patch_mask), out)
