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
@pytest.mark.parametrize(
    ("schedule", "dtype", "clipped_lambda"),
    [
        ("quad", torch.float32, -5.1),
        ("quad", torch.float64, -5.1),
        ("linear", torch.float32, -5.1),
        ("linear", torch.float32, -2.0),
    ],
)
def test_noise_schedule(
    device: torch.device,
    schedule: Literal["quad", "linear"],
    dtype: torch.dtype,
    clipped_lambda: float,
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
    module = NoiseScheduleVP(discrete.beta, clipped_lambda=clipped_lambda)
    log_alpha = 0.5 * (1 - discrete.beta).log().cumsum(0)
    half_log_snr = log_alpha - 0.5 * (1 - (2 * log_alpha).exp()).log()
    retained = int((half_log_snr >= clipped_lambda).sum())
    assert module.total_N == retained
    assert (retained < 1000) == (schedule == "quad" or clipped_lambda == -2.0)

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
    """Preserve boundary extrapolation and query-time gradients."""
    betas = torch.tensor(
        data=[0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float64
    )
    module = NoiseScheduleVP(betas)
    logs = 0.5 * (1 - betas).log().cumsum(0)
    t = torch.tensor(
        data=[[0.2, 0.25], [0.375, 1.1]],
        device=device,
        dtype=betas.dtype,
        requires_grad=True,
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
    (gradient,) = torch.autograd.grad(
        module.marginal_log_mean_coeff(t).sum(), t
    )
    assert torch.isfinite(gradient).all()
    assert (gradient != 0).any()


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


def test_noise_schedule_too_short() -> None:
    """Reject schedules without two usable interpolation knots."""
    with pytest.raises(ValueError, match="At least two diffusion steps"):
        NoiseScheduleVP(torch.tensor([0.99999, 0.99999]))
