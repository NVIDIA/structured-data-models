# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models import KumoAnomaly
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(device=device)
    reference = torch.randn(50, 3, device=device)
    x = torch.randn(130, 3, device=device)

    def generator() -> torch.Generator:
        return torch.Generator(device).manual_seed(0)

    out = model(
        x,
        reference,
        num_samples=2,
        num_steps=2,
        generator=generator(),
    )
    assert out.size() == (130, 1)
    assert out.columns["numerical"] == ("score",)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert (out.numerical >= 0).all()

    again = model(
        x,
        reference,
        num_samples=2,
        num_steps=2,
        generator=generator(),
    )
    assert torch.equal(again.numerical, out.numerical)


def test_transductive_forward() -> None:
    model = KumoAnomaly()
    x = torch.randn(30, 2, dtype=torch.float64)

    out = model(x, num_samples=2, num_steps=2)
    assert out.size() == (30, 1)
    assert out.numerical.isfinite().all()


def test_too_many_features() -> None:
    model = KumoAnomaly()
    x = torch.randn(10, 41)

    with pytest.raises(ValueError, match="at most 40 features"):
        model(x)
