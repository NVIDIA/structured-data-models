# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import pytest
import torch

from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("schedule", "num_steps", "beta_start", "beta_end", "dtype"),
    [
        ("quad", 1000, 1e-4, 0.5, torch.float64),
        ("linear", 1000, 1e-4, 0.02, torch.float64),
        ("quad", 1000, 1e-4, 0.5, torch.float32),
        ("quad", 1, 1e-4, 0.5, torch.float64),
        ("linear", 100, 1e-10, 0.9, torch.float64),
    ],
)
def test_diffusion_schedule(
    device: torch.device,
    schedule: Literal["quad", "linear"],
    num_steps: int,
    beta_start: float,
    beta_end: float,
    dtype: torch.dtype,
) -> None:
    module = DiffusionSchedule(
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
        device=device,
        dtype=dtype,
    )
    if schedule == "quad":
        beta = np.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
    else:
        beta = np.linspace(beta_start, beta_end, num_steps)
    alpha_hat = 1 - beta
    alpha = np.clip(np.cumprod(alpha_hat), 1e-6, 1 - 1e-6)
    for actual, values in (
        (module.beta, beta),
        (module.alpha_hat, alpha_hat),
        (module.alpha, alpha),
    ):
        expected = torch.tensor(values, device=device, dtype=dtype)
        assert actual.shape == (num_steps,)
        assert actual.dtype == dtype
        assert actual.device == device
        assert not actual.requires_grad
        torch.testing.assert_close(
            actual=actual,
            expected=expected,
            rtol=1e-12 if dtype == torch.float64 else 1e-6,
            atol=1e-15,
        )

    # Unsorted and repeated indices retain their layout, including endpoints.
    step = torch.tensor(
        [[num_steps - 1, 0], [num_steps // 2, 0]], device=device
    )
    expected = torch.tensor(alpha, device=device, dtype=dtype)[step]
    torch.testing.assert_close(module(step), expected)
    if beta_start < 1e-6:
        assert module.alpha[0] == 1 - 1e-6
        assert module.alpha[-1] == 1e-6
        assert module.alpha_hat[0] > 1 - 1e-6


@withCUDA
def test_diffusion_schedule_conversion(device: torch.device) -> None:
    module = DiffusionSchedule(num_steps=10, beta_start=1e-4, beta_end=0.5)
    assert module.beta.dtype == torch.float64
    original = [module.beta, module.alpha_hat, module.alpha]
    module = module.to(device=device, dtype=torch.float32)
    for actual, expected in zip(
        (module.beta, module.alpha_hat, module.alpha), original, strict=True
    ):
        torch.testing.assert_close(actual, expected.to(device).float())
    assert not module.state_dict()


def test_diffusion_schedule_default_device() -> None:
    with torch.device("meta"):
        module = DiffusionSchedule(num_steps=3, beta_start=0.1, beta_end=0.3)
        cpu_module = DiffusionSchedule(
            num_steps=3, beta_start=0.1, beta_end=0.3, device="cpu"
        )
    assert all(buffer.device.type == "meta" for buffer in module.buffers())
    assert all(buffer.device.type == "cpu" for buffer in cpu_module.buffers())
