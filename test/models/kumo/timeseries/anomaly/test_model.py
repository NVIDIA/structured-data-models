# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models import KumoAnomaly
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(pretrained=False, device=device)
    x = torch.randn(2, 130, 3, device=device)

    def generator() -> torch.Generator:
        return torch.Generator(device).manual_seed(0)

    out = model(x, num_samples=1, num_steps=1, generator=generator())
    assert out.size() == (2, 130, 1)
    assert out.columns["numerical"] == ("mae",)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert (out.numerical >= 0).all()

    again = model(x, num_samples=1, num_steps=1, generator=generator())
    assert torch.equal(again.numerical, out.numerical)
