# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from torch.nn import ReLU

from sdm.nn import ResidualMLP
from sdm.testing import withCUDA


@withCUDA
def test_residual_mlp(device: torch.device) -> None:
    block = ResidualMLP(
        in_channels=3,
        hidden_channels=2,
        out_channels=2,
        activation=ReLU(),
        bias=False,
        device=device,
    )

    out = block(torch.randn(1, 3, device=device))
    assert out.size() == (1, 2)
