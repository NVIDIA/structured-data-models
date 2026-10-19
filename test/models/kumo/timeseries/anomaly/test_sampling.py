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


@withCUDA
@pytest.mark.parametrize(
    ("num_steps", "expected"),
    [
        (1, [-3.194379, -1.927232, -0.738264, 0.413303, 1.602271, 2.869418]),
        (2, [-3.047740, -1.722190, -0.563134, 0.399716, 1.515242, 2.826252]),
        (3, [-3.082448, -1.776467, -0.616305, 0.404998, 1.538929, 2.834348]),
        (20, [-3.038120, -1.723861, -0.578480, 0.406552, 1.523009, 2.825194]),
    ],
)
def test_dpm_solver_reference(
    device: torch.device,
    num_steps: int,
    expected: list[float],
) -> None:
    """Match the reference DPM-Solver++ with training step indices."""
    # Generated with Kumo-TS af74798, utils/dpm_solver_pytorch.py,
    # DPM_Solver(algorithm_type="dpmsolver++").sample(order=2,
    # method="singlestep", skip_type="time_uniform"), with a model function
    # that maps each time t to the step index round(t * total_N - 1).
    solver = DPMSolver(betas=BETAS, num_steps=num_steps).to(device)
    x = torch.linspace(-1, 1, 6, device=device).reshape(1, 2, 3)

    def predict_noise(sample: Tensor, step: Tensor) -> Tensor:
        return 0.2 * sample.tanh() + 0.1 * (step / 1000).sin()

    out = solver(x, predict_noise)
    assert out.shape == x.shape
    torch.testing.assert_close(
        actual=out.flatten(),
        expected=torch.tensor(expected, device=device),
    )


def test_dpm_solver_steps() -> None:
    """Pass decreasing training step indices, including midpoints."""
    steps = []

    def predict_noise(sample: Tensor, step: Tensor) -> Tensor:
        steps.append(step)
        return torch.zeros_like(sample)

    DPMSolver(BETAS, num_steps=20)(torch.zeros(1, 2, 4), predict_noise)
    assert torch.cat(steps).tolist() == [
        499, 474, 449, 424, 399, 374, 349, 324, 299, 274,
        250, 225, 200, 175, 150, 125, 100, 75, 50, 25,
    ]  # fmt: skip


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_steps", [1, 2, 3, 20])
def test_dpm_solver_constant_data(
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int,
) -> None:
    """Reproduce a constant clean-data prediction for either order."""
    # A constant clean signal and noise define the exact ODE trajectory
    # alpha_t * clean + sigma_t * noise, which the solver follows exactly.
    schedule = NoiseScheduleVP(BETAS)
    time = torch.tensor([1.0, 1 / schedule.total_N])
    alpha_start, alpha_end = schedule.marginal_alpha(time).tolist()
    sigma_start, sigma_end = schedule.marginal_std(time).tolist()
    clean = torch.full((2, 3, 4), 0.25, device=device, dtype=dtype)
    noise = torch.randn_like(clean)
    x = alpha_start * clean + sigma_start * noise
    solver = DPMSolver(BETAS, num_steps=num_steps).to(device)

    out = solver(x, lambda sample, step: noise)
    assert out.device == device
    assert out.dtype == dtype
    expected = alpha_end * clean + sigma_end * noise
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_dpm_solver_num_steps() -> None:
    """Reject solvers that would return the initial noise unchanged."""
    with pytest.raises(ValueError, match="num_steps"):
        DPMSolver(BETAS, num_steps=0)
