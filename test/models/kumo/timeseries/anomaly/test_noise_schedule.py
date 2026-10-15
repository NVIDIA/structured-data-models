# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np
import pytest
import torch

from sdm.models.kumo.timeseries.anomaly.noise_schedule import NoiseScheduleVP
from sdm.models.kumo.timeseries.anomaly.schedule import DiffusionSchedule
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("schedule", ["quad", "linear"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noise_schedule(
    device: torch.device,
    schedule: Literal["quad", "linear"],
    dtype: torch.dtype,
) -> None:
    """Match VP coefficients, tail truncation, and interpolation."""
    discrete = DiffusionSchedule(
        num_steps=1000,
        beta_start=1e-4,
        beta_end=0.5 if schedule == "quad" else 0.02,
        schedule=schedule,
        dtype=dtype,
        device=device,
    )
    module = NoiseScheduleVP(discrete.beta)
    log_alpha = 0.5 * (1 - discrete.beta).log().cumsum(0)
    half_log_snr = log_alpha - 0.5 * (1 - (2 * log_alpha).exp()).log()
    retained = int((half_log_snr >= -5.1).sum())
    assert module.total_N == retained
    assert module.T == 1.0
    assert module.schedule == "discrete"
    assert (retained < 1000) == (schedule == "quad")

    grid = torch.linspace(0, 1, retained + 1, dtype=torch.float32)[1:].to(
        discrete.beta
    )
    torch.testing.assert_close(
        actual=module.marginal_log_mean_coeff(grid),
        expected=log_alpha[:retained],
    )
    # Unsorted queries, repeated values, and both interval endpoints.
    t = torch.tensor(
        data=[1.0, 0.435, 1 / retained, 0.731, 0.435],
        device=device,
        dtype=dtype,
    )
    expected = torch.tensor(
        data=np.interp(
            x=t.cpu().numpy(),
            xp=grid.cpu().numpy(),
            fp=log_alpha[:retained].cpu().numpy(),
        ),
        device=device,
        dtype=dtype,
    )
    alpha = expected.exp()
    sigma = (1 - (2 * expected).exp()).sqrt()
    for actual, target in (
        (module.marginal_log_mean_coeff(t), expected),
        (module.marginal_alpha(t), alpha),
        (module.marginal_std(t), sigma),
        (module.marginal_lambda(t), expected - sigma.log()),
        (module.inverse_lambda(module.marginal_lambda(t)), t),
    ):
        assert actual.device == device
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, target)


@withCUDA
def test_noise_schedule_extrapolation(device: torch.device) -> None:
    """Extrapolate boundary segments and flatten scalar/batched inputs."""
    betas = torch.tensor(
        data=[0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float64
    )
    module = NoiseScheduleVP(betas)
    logs = 0.5 * (1 - betas).log().cumsum(0)
    t = torch.tensor(
        data=[[0.2, 0.25], [0.375, 1.1]], device=device, dtype=betas.dtype
    )
    expected = torch.stack(
        tensors=[
            logs[0] - 0.2 * (logs[1] - logs[0]),
            logs[0],
            (logs[0] + logs[1]) / 2,
            logs[-1] + 0.4 * (logs[-1] - logs[-2]),
        ],
    )
    torch.testing.assert_close(module.marginal_log_mean_coeff(t), expected)
    torch.testing.assert_close(
        actual=module.inverse_lambda(module.marginal_lambda(t)),
        expected=t.flatten(),
    )
    torch.testing.assert_close(
        actual=module.marginal_alpha(t[0, 0]), expected=expected[:1].exp()
    )
    empty = torch.empty(0, device=device, dtype=betas.dtype)
    assert module.marginal_alpha(empty).shape == (0,)
    assert module.inverse_lambda(empty).shape == (0,)


@withCUDA
def test_noise_schedule_conversion(device: torch.device) -> None:
    """Move coefficients while keeping outputs and checkpoint contents."""
    betas = torch.tensor(
        data=[0.1, 0.2, 0.3, 0.4], dtype=torch.float64, requires_grad=True
    )
    module = NoiseScheduleVP(betas)
    t = torch.tensor([0.3, 0.6, 0.9], dtype=betas.dtype)
    expected = module.marginal_lambda(t).detach()
    module = module.to(device=device, dtype=torch.float32)
    actual = module.marginal_lambda(t.to(device).float())
    torch.testing.assert_close(actual, expected.to(device).float())
    assert not actual.requires_grad
    assert not module.state_dict()


@withCUDA
def test_noise_schedule_gradients(device: torch.device) -> None:
    """Differentiate time and half-logSNR away from interpolation knots."""
    betas = torch.tensor(
        data=[0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float64
    )
    module = NoiseScheduleVP(betas)
    t = torch.tensor(
        data=[0.3, 0.6, 0.9],
        device=device,
        dtype=betas.dtype,
        requires_grad=True,
    )
    for method in (
        module.marginal_log_mean_coeff,
        module.marginal_alpha,
        module.marginal_std,
        module.marginal_lambda,
    ):
        assert torch.autograd.gradcheck(method, (t,))
    lamb = module.marginal_lambda(t).detach().requires_grad_()
    assert torch.autograd.gradcheck(module.inverse_lambda, (lamb,))


@pytest.mark.parametrize("betas", [[], [0.1], [0.99999, 0.99999]])
def test_noise_schedule_too_short(betas: list[float]) -> None:
    """Reject schedules without two usable interpolation knots."""
    with pytest.raises(ValueError, match="At least two diffusion steps"):
        NoiseScheduleVP(torch.tensor(betas))
