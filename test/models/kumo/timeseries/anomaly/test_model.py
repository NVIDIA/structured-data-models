# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models import KumoAnomaly
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(pretrained=False, device=device)
    x = TableTensor.from_tensor(torch.randn(32, 5, device=device))

    out = model(x)
    assert out.size() == (32, 1)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert "mae" in out.numerical_columns
