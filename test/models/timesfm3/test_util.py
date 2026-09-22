# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.models.timesfm3.util import (
    get_running_stats,
    update_running_stats,
)
from sdm.testing import withCUDA


@withCUDA
def test_update_running_stats(device: torch.device) -> None:
    x = torch.tensor(
        [[[1.0, 100.0, 2.0, 3.0]]],
        device=device,
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[[False, True, False, False]]],
        device=device,
    )
    initial = torch.zeros(1, 1, device=device, dtype=x.dtype)

    n, mu, sigma = update_running_stats(
        initial,
        initial,
        initial,
        x,
        mask,
    )

    assert n.dtype == x.dtype
    assert n.device == device
    torch.testing.assert_close(n, n.new_tensor([[3.0]]))
    torch.testing.assert_close(mu, mu.new_tensor([[2.0]]))
    torch.testing.assert_close(
        sigma,
        sigma.new_tensor([[2.0 / 3.0]]).sqrt(),
    )


@withCUDA
def test_update_running_stats_preserves_upstream_operation_order(
    device: torch.device,
) -> None:
    n = torch.tensor([[11.0]], device=device)
    mu = torch.tensor([[-92.03504943847656]], device=device)
    sigma = torch.tensor([[6.244617462158203]], device=device)
    x = torch.tensor(
        [[[-17.37119483947754, -76.65009307861328, 79.12385559082031]]],
        device=device,
    )
    mask = torch.zeros_like(x, dtype=torch.bool)

    _, actual_mu, actual_sigma = update_running_stats(n, mu, sigma, x, mask)

    inc_n = (~mask).float().sum(dim=-1)
    inc_mu = x.mean(dim=-1)
    inc_sigma = ((x - inc_mu.unsqueeze(-1)).square().mean(dim=-1)).sqrt()
    new_n = n + inc_n
    expected_sigma = (
        (
            n * sigma * sigma
            + inc_n * inc_sigma * inc_sigma
            + n * (mu - actual_mu) * (mu - actual_mu)
            + inc_n * (inc_mu - actual_mu) * (inc_mu - actual_mu)
        )
        / new_n
    ).sqrt()
    torch.testing.assert_close(
        actual_sigma,
        expected_sigma,
        rtol=0.0,
        atol=0.0,
    )


@withCUDA
def test_update_running_stats_fully_masked(device: torch.device) -> None:
    n = torch.tensor([[3.0]], device=device)
    mu = torch.tensor([[2.0]], device=device)
    sigma = torch.tensor([[1.0]], device=device)
    x = torch.tensor([[[100.0, 200.0]]], device=device)
    mask = torch.ones_like(x, dtype=torch.bool)

    actual = update_running_stats(n, mu, sigma, x, mask)

    for output, expected in zip(actual, (n, mu, sigma), strict=True):
        torch.testing.assert_close(output, expected)


@withCUDA
def test_get_running_stats_uses_float32_for_bfloat16(
    device: torch.device,
) -> None:
    values = torch.ones(1, 1, 10, 1000, dtype=torch.bfloat16, device=device)
    masks = torch.zeros_like(values, dtype=torch.bool)

    n, mu, sigma = get_running_stats(values, masks)

    assert n.dtype == torch.float32
    assert mu.dtype == torch.float32
    assert sigma.dtype == torch.float32
    torch.testing.assert_close(n[..., -1], n.new_tensor([[10_000.0]]))
    torch.testing.assert_close(mu[..., -1], mu.new_tensor([[1.0]]))
    torch.testing.assert_close(sigma[..., -1], sigma.new_tensor([[0.0]]))


@withCUDA
def test_get_running_stats_accumulates_across_patches(
    device: torch.device,
) -> None:
    values = torch.tensor(
        [[[[1.0, 3.0], [5.0, 99.0], [7.0, 9.0]]]],
        device=device,
    )
    masks = torch.tensor(
        [[[[False, False], [False, True], [True, False]]]],
        device=device,
    )

    n, mu, sigma = get_running_stats(values, masks)

    torch.testing.assert_close(n, n.new_tensor([[[2.0, 3.0, 4.0]]]))
    torch.testing.assert_close(mu, mu.new_tensor([[[2.0, 3.0, 4.5]]]))
    torch.testing.assert_close(
        sigma,
        sigma.new_tensor([[[1.0, math.sqrt(8.0 / 3.0), math.sqrt(8.75)]]]),
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_get_running_stats_has_finite_zero_variance_gradients(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    values = torch.tensor(
        [[[[9.0, 10.0], [2.0, 2.0], [9.0, 10.0]]]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    masks = torch.tensor(
        [[[[True, True], [False, False], [True, True]]]],
        device=device,
    )

    _, mu, sigma = get_running_stats(values, masks)
    (mu + sigma).sum().backward()

    assert values.grad is not None
    torch.testing.assert_close(
        values.grad,
        values.grad.new_tensor([[[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]]),
    )
