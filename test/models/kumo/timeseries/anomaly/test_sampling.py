# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import pytest
import torch

from sdm.models.kumo.timeseries.anomaly.sampling import DDPMStep
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("schedule", "num_steps", "indices", "dtype"),
    [
        ("quad", 7, [6, 0, 3, 3], torch.float32),
        ("linear", 7, [3], torch.float64),
        ("quad", 1, [0], torch.float32),
        ("linear", 12, [11, 2, 2, 0], torch.float16),
        ("quad", 6, [5], torch.bfloat16),
        ("quad", 100, [99, 0, 50, 99], torch.float32),
    ],
)
def test_ddpm_step(
    device: torch.device,
    schedule: Literal["quad", "linear"],
    num_steps: int,
    indices: list[int],
    dtype: torch.dtype,
) -> None:
    """Match Kumo-TS reverse coefficients for shared and per-window steps."""
    coefficients = DiffusionSchedule(
        num_steps=num_steps,
        beta_start=1e-4,
        beta_end=0.5,
        schedule=schedule,
        device=device,
    )
    module = DDPMStep(coefficients)
    x = torch.randn(4, 2, 5, device=device, dtype=dtype)
    predicted = torch.randn_like(x)
    noise = torch.randn_like(x)
    step = torch.tensor(indices, device=device)
    out = module(x=x, predicted_noise=predicted, step=step, noise=noise)
    if schedule == "quad":
        beta = np.linspace(1e-4**0.5, 0.5**0.5, num_steps) ** 2
    else:
        beta = np.linspace(1e-4, 0.5, num_steps)
    alpha_hat = 1 - beta
    alpha = np.clip(np.cumprod(alpha_hat), 1e-6, 1 - 1e-6)
    expected = torch.empty_like(x)
    for batch in range(x.size(0)):
        t = indices[batch] if len(indices) > 1 else indices[0]
        coeff1 = x.new_tensor(1 / alpha_hat[t] ** 0.5)
        coeff2 = x.new_tensor((1 - alpha_hat[t]) / (1 - alpha[t]) ** 0.5)
        expected[batch] = coeff1 * (x[batch] - coeff2 * predicted[batch])
        if t > 0:
            sigma = ((1 - alpha[t - 1]) / (1 - alpha[t]) * beta[t]) ** 0.5
            expected[batch] += x.new_tensor(sigma) * noise[batch]
    torch.testing.assert_close(out, expected)
    assert out.dtype == dtype
    assert out.device == device


@withCUDA
def test_ddpm_step_final_and_gradients(device: torch.device) -> None:
    """Ignore noise at the final step while retaining sample gradients."""
    schedule = DiffusionSchedule(
        num_steps=3, beta_start=0.1, beta_end=0.3, device=device
    )
    module = DDPMStep(schedule)
    x = torch.randn(
        size=(2, 1, 4), device=device, dtype=torch.float64, requires_grad=True
    )
    predicted = torch.randn_like(x, requires_grad=True)
    noise = torch.randn_like(x, requires_grad=True)
    step = torch.tensor(0, device=device)
    out = module(x=x, predicted_noise=predicted, step=step, noise=noise)
    other = module(
        x=x, predicted_noise=predicted, step=step, noise=noise * 100
    )
    torch.testing.assert_close(out, other)
    out.sum().backward()
    coeff1 = schedule.alpha_hat[0].rsqrt()
    coeff2 = (1 - schedule.alpha_hat[0]) / (1 - schedule.alpha[0]).sqrt()
    torch.testing.assert_close(x.grad, coeff1.expand_as(x))
    torch.testing.assert_close(predicted.grad, (-coeff1 * coeff2).expand_as(x))
    torch.testing.assert_close(noise.grad, torch.zeros_like(noise))


@withCUDA
def test_ddpm_step_conversion(device: torch.device) -> None:
    """Move derived coefficients with the module without checkpoint entries."""
    schedule = DiffusionSchedule(num_steps=3, beta_start=0.1, beta_end=0.3)
    module = DDPMStep(schedule)
    x = torch.randn(2, 1, 4)
    predicted = torch.randn_like(x)
    noise = torch.randn_like(x)
    step = torch.tensor([2, 1])
    expected = module(x=x, predicted_noise=predicted, step=step, noise=noise)
    module.to(device=device, dtype=torch.float32)
    actual = module(
        x=x.to(device),
        predicted_noise=predicted.to(device),
        step=step.to(device),
        noise=noise.to(device),
    )
    torch.testing.assert_close(actual, expected.to(device))
    assert not module.state_dict()
