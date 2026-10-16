# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.noise_schedule import NoiseScheduleVP
from sdm.models.kumo.timeseries.anomaly.sampling import DPMSolver
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("num_steps", "expected"),
    [
        (
            1,
            [
                -3.19437909,
                -1.92723203,
                -0.73826426,
                0.41330332,
                1.60227108,
                2.86941814,
            ],
        ),
        (
            2,
            [
                -3.04774046,
                -1.72219038,
                -0.56313372,
                0.39971635,
                1.51524186,
                2.82625222,
            ],
        ),
        (
            3,
            [
                -3.08244777,
                -1.77646685,
                -0.61630511,
                0.40499809,
                1.53892899,
                2.83434820,
            ],
        ),
        (
            20,
            [
                -3.03812027,
                -1.72386146,
                -0.57848012,
                0.40655187,
                1.52300930,
                2.82519412,
            ],
        ),
    ],
)
def test_nonlinear_reference(
    device: torch.device, num_steps: int, expected: list[float]
) -> None:
    """Match the pinned solver with the corrected model adapter."""
    # Generated with Kumo-TS af74798bbf7d020d3298f545fb9be383560a18ce,
    # utils/dpm_solver_pytorch.py, DPM_Solver.sample(order=2,
    # method="singlestep", skip_type="time_uniform"), using noise() below
    # and round(t * total_N - 1) as the continuous-time model adapter.
    betas = DiffusionSchedule(500, 0.0001, 0.01, "linear").beta.float()
    solver = DPMSolver(betas=betas, num_steps=num_steps).to(device)
    x = torch.linspace(-1, 1, 6, device=device).reshape(1, 2, 3)

    def noise(sample: Tensor, step: Tensor) -> Tensor:
        """Make both the evolving state and the trained timestep observable."""
        return 0.2 * sample.tanh() + 0.1 * (step / 1000).sin().reshape(
            -1, 1, 1
        )

    torch.testing.assert_close(
        actual=solver(x, noise).flatten(),
        expected=torch.tensor(expected, device=device),
    )


@withCUDA
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10, 20, 21])
def test_constant_clean_prediction(
    device: torch.device, num_steps: int
) -> None:
    """Both solver orders integrate a constant clean-data prediction."""
    betas = torch.linspace(0.0001, 0.01, 500)
    solver = DPMSolver(betas=betas, num_steps=num_steps).to(device)
    x = torch.randn(2, 3, 4, device=device)
    clean = torch.full_like(x, 0.25)
    calls = []
    coefficients = []
    for i, order in enumerate(solver.orders):
        coefficients.extend(
            [(solver.alpha[i, j], solver.sigma[i, j]) for j in range(order)]
        )

    def noise(sample: Tensor, step: Tensor) -> Tensor:
        """Represent a known clean prediction as noise at each solver time."""
        alpha, sigma = coefficients[len(calls)]
        calls.append(step)
        return (sample - alpha * clean) / sigma

    actual = solver(x, noise)
    schedule = NoiseScheduleVP(betas).to(device)
    end = torch.tensor([1 / schedule.total_N], device=device)
    alpha_end, sigma_end = (
        schedule.marginal_alpha(end),
        schedule.marginal_std(end),
    )
    sigma_start = schedule.marginal_std(torch.ones_like(end))
    alpha_start = schedule.marginal_alpha(torch.ones_like(end))
    expected = alpha_end * clean + sigma_end / sigma_start * (
        x - alpha_start * clean
    )
    torch.testing.assert_close(actual, expected)
    assert len(calls) == num_steps
    assert solver.state_dict() == {}


def test_timestep_mapping() -> None:
    """The default path visits decreasing noise levels instead of all 499."""
    solver = DPMSolver(torch.linspace(0.0001, 0.01, 500))
    steps = []

    def noise(x: Tensor, step: Tensor) -> Tensor:
        """Expose the actual integer indices passed to the denoiser."""
        steps.append(step.clone())
        return torch.zeros_like(x)

    solver(torch.zeros(1, 2, 4), noise)
    actual = torch.cat(steps)
    expected = (torch.linspace(1, 1 / 500, 21)[:-1] * 500 - 1).round().long()
    torch.testing.assert_close(actual, expected)
    assert actual[0] == 499
    assert actual[-1] == 25
    assert (actual.diff() < 0).all()


def test_truncated_schedule_indices() -> None:
    """Clipped schedules keep the training indices of retained knots."""
    betas = torch.linspace(0.001, 0.2, 1000)
    schedule = NoiseScheduleVP(betas)
    assert schedule.total_N < len(betas)
    solver = DPMSolver(betas, num_steps=3)
    assert solver.steps[0, 0] == schedule.total_N - 1
    assert solver.steps[-1, -1] == 0


def test_solver_nonempty() -> None:
    """A zero-evaluation solver must not return unmodified noise as data."""
    with pytest.raises(ValueError, match="num_steps"):
        DPMSolver(torch.linspace(0.0001, 0.01, 500), num_steps=0)
