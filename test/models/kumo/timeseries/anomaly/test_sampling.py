# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.noise_schedule import NoiseScheduleVP
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.testing import withCUDA

BETAS = DiffusionSchedule(500, 0.0001, 0.01, "linear").beta.float()


def test_dpm_solver_steps() -> None:
    steps = []

    def predict_noise(sample: Tensor, step: Tensor) -> Tensor:
        steps.append(step)
        return torch.zeros_like(sample)

    DPMSolver(BETAS, num_steps=3)(torch.zeros(2, 3), predict_noise)
    torch.testing.assert_close(torch.cat(steps), torch.tensor([499, 333, 166]))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dpm_solver_constant_data(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    # A constant clean signal and noise define the exact ODE trajectory
    # alpha_t * clean + sigma_t * noise, which the solver follows exactly.
    schedule = NoiseScheduleVP(BETAS)
    time = torch.tensor([1.0, 1 / schedule.total_N])
    alpha_start, alpha_end = schedule.marginal_alpha(time).tolist()
    sigma_start, sigma_end = schedule.marginal_std(time).tolist()
    clean = torch.full((2, 3, 4), 0.25, device=device, dtype=dtype)
    noise = torch.randn_like(clean)
    x = alpha_start * clean + sigma_start * noise
    # Three steps take one second-order and one first-order update.
    solver = DPMSolver(BETAS, num_steps=3).to(device)

    def predict_noise(sample: Tensor, step: Tensor) -> Tensor:
        assert step.device == device
        return noise

    out = solver(x, predict_noise)
    expected = alpha_end * clean + sigma_end * noise
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_dpm_solver_linear_noise() -> None:
    # Predicting the sample itself as noise has the exact solution
    # x_end / x_start = (alpha_end / alpha_start) * exp(u_end - u_start),
    # where u = asinh(sigma / alpha). The solver stays within 1%, while
    # first-order updates alone miss it by 5%.
    schedule = NoiseScheduleVP(BETAS)
    time = torch.tensor([1.0, 1 / schedule.total_N])
    alpha = schedule.marginal_alpha(time)
    u = (schedule.marginal_std(time) / alpha).asinh()
    expected = (alpha[1] / alpha[0]) * (u[1] - u[0]).exp()

    out = DPMSolver(BETAS, num_steps=21)(torch.ones(1), lambda x, step: x)
    torch.testing.assert_close(out, expected.view(1), rtol=1e-2, atol=0.0)
