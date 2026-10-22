# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.nn import ResidualMLP
from sdm.testing import withCUDA


@withCUDA
def test_residual_mlp(device: torch.device) -> None:
    model = ResidualMLP(3, 7, 5, device=device)
    x = torch.randn(2, 4, 3, device=device, requires_grad=True)
    out = model(x)
    assert out.size() == (2, 4, 5)
    out.square().sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
