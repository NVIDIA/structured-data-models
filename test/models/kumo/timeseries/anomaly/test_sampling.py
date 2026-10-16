# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import pytest
import torch
from torch import Tensor

from sdm.models.kumo.timeseries.anomaly.sampling import dpm_solver_update
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_dpm_solver_update(
    device: torch.device,
    dtype: torch.dtype,
    second_order: bool,
    batched: bool,
) -> None:
    """Match the reference formulas with shared or per-window coefficients."""
    x = torch.randn(2, 3, 5, device=device, dtype=dtype)
    data_start = torch.randn_like(x)
    data_mid = torch.randn_like(x)
    # Each row is alpha_end, sigma_start, sigma_end, lambda_delta, r1.
    values = x.new_tensor(
        data=[[0.8, 0.9, 0.6, 0.6, 0.25], [0.6, 0.95, 0.8, 0.2, 0.75]]
    )
    if batched:
        alpha, sigma_start, sigma_end, delta, r1 = values.T[:, :, None, None]
    else:
        alpha, sigma_start, sigma_end, delta, r1 = values[0]
    before = [value.clone() for value in (x, data_start, data_mid, values)]
    actual = dpm_solver_update(
        x=x,
        data_start=data_start,
        alpha_end=alpha,
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        lambda_delta=delta,
        data_mid=data_mid if second_order else None,
        r1=r1,
    )
    expected = torch.empty_like(x)
    for i in range(x.size(0)):
        a, s, t, h, r = values[i if batched else 0].tolist()
        factor = a * math.expm1(-h)
        expected[i] = (t / s) * x[i] - factor * data_start[i]
        if second_order:
            expected[i] -= (0.5 / r) * factor * (data_mid[i] - data_start[i])
    torch.testing.assert_close(actual, expected)
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    assert actual.device == device
    for value, original in zip(
        (x, data_start, data_mid, values), before, strict=True
    ):
        torch.testing.assert_close(value, original)


@withCUDA
def test_dpm_solver_update_limits(device: torch.device) -> None:
    """Check constant predictions, zero intervals, and midpoint limits."""
    x = torch.randn(2, 3, 5, device=device)
    predicted = torch.randn_like(x)
    update = partial(
        dpm_solver_update,
        x=x,
        data_start=predicted,
        alpha_end=x.new_tensor(0.8),
        sigma_start=x.new_tensor(0.6),
        sigma_end=x.new_tensor(0.6),
    )
    delta = x.new_tensor(0.3)
    torch.testing.assert_close(
        actual=update(lambda_delta=delta, data_mid=predicted, r1=0.3),
        expected=update(lambda_delta=delta),
    )
    torch.testing.assert_close(
        actual=update(lambda_delta=x.new_zeros(()), data_mid=predicted + 1),
        expected=x,
        rtol=0,
        atol=0,
    )
    # A half-logSNR midpoint cancels the start prediction in the final step.
    mid = predicted + 1
    torch.testing.assert_close(
        actual=update(lambda_delta=delta, data_mid=mid),
        expected=x - 0.8 * (-delta).expm1() * mid,
    )


@withCUDA
@pytest.mark.parametrize("second_order", [False, True])
def test_dpm_solver_update_small_interval(
    device: torch.device,
    second_order: bool,
) -> None:
    """Retain updates too small for an exp-minus-one computation in float32."""
    x = torch.zeros(2, 1, 3, device=device)
    out = dpm_solver_update(
        x=x,
        data_start=torch.ones_like(x),
        alpha_end=x.new_tensor(1.0),
        sigma_start=x.new_tensor(1.0),
        sigma_end=x.new_tensor(1.0),
        lambda_delta=x.new_tensor(1e-10),
        data_mid=torch.full_like(x, 3) if second_order else None,
        r1=0.25,
    )
    expected = torch.full_like(x, 5e-10 if second_order else 1e-10)
    torch.testing.assert_close(
        actual=out, expected=expected, rtol=1e-6, atol=0
    )


@withCUDA
@pytest.mark.parametrize("second_order", [False, True])
def test_dpm_solver_update_gradients(
    device: torch.device,
    second_order: bool,
) -> None:
    """Differentiate samples and clean-data predictions for either order."""
    inputs = tuple(
        torch.randn(
            size=(2, 1, 2),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        for _ in range(3)
    )

    def evaluate(x: Tensor, start: Tensor, mid: Tensor) -> Tensor:
        """Evaluate with fixed coefficients for gradient checking."""
        return dpm_solver_update(
            x=x,
            data_start=start,
            alpha_end=x.new_tensor(0.8),
            sigma_start=x.new_tensor(0.9),
            sigma_end=x.new_tensor(0.6),
            lambda_delta=x.new_tensor(0.6),
            data_mid=mid if second_order else None,
            r1=0.3,
        )

    assert torch.autograd.gradcheck(evaluate, inputs)
