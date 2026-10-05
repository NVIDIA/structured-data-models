# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.models.timesfm3.util import (
    get_running_stats,
    revin,
    update_running_stats,
)
from sdm.testing import withCUDA


@withCUDA
def test_update_running_stats(device: torch.device) -> None:
    x = torch.tensor([[[1.0, 100.0, 2.0, 3.0]]], device=device)
    mask = torch.tensor([[[False, True, False, False]]], device=device)

    count, mean, std = update_running_stats(
        count=torch.zeros(1, 1, device=device, dtype=torch.int64),
        mean=torch.zeros(1, 1, device=device, dtype=x.dtype),
        std=torch.zeros(1, 1, device=device, dtype=x.dtype),
        x=x,
        mask=mask,
    )

    torch.testing.assert_close(count, count.new_tensor([[3]]))
    torch.testing.assert_close(mean, mean.new_tensor([[2.0]]))
    torch.testing.assert_close(std, std.new_tensor([[2.0 / 3.0]]).sqrt())


@withCUDA
def test_update_running_stats_fully_masked(device: torch.device) -> None:
    count = torch.tensor([[3]], device=device)
    mean = torch.tensor([[2.0]], device=device)
    std = torch.tensor([[1.0]], device=device)
    x = torch.tensor([[[100.0, 200.0]]], device=device)
    mask = torch.ones_like(x, dtype=torch.bool)

    actual = update_running_stats(count, mean, std, x, mask)

    for output, expected in zip(actual, (count, mean, std)):
        torch.testing.assert_close(output, expected)


@withCUDA
def test_get_running_stats(
    device: torch.device,
) -> None:
    x = torch.tensor([[[[1.0, 3.0], [5.0, 99.0], [7.0, 9.0]]]], device=device)
    mask = torch.tensor(
        [[[[False, False], [False, True], [True, False]]]],
        device=device,
    )
    
    count, mean, std = get_running_stats(x, mask)

    torch.testing.assert_close(count, count.new_tensor([[[2, 3, 4]]]))
    torch.testing.assert_close(mean, mean.new_tensor([[[2.0, 3.0, 4.5]]]))
    torch.testing.assert_close(
        std,
        std.new_tensor([[[1.0, math.sqrt(8.0 / 3.0), math.sqrt(8.75)]]]),
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


@withCUDA
def test_revin_round_trip(device: torch.device) -> None:
    x = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        device=device,
        dtype=torch.float64,
    )
    mu = torch.tensor([[2.0]], device=device, dtype=x.dtype)
    sigma = torch.tensor([[0.5]], device=device, dtype=x.dtype)

    normalized = revin(x, mu, sigma)
    restored = revin(normalized, mu, sigma, reverse=True)

    assert normalized.dtype == x.dtype
    assert normalized.device == device
    torch.testing.assert_close(restored, x)


@withCUDA
def test_revin_near_zero_sigma(device: torch.device) -> None:
    x = torch.tensor([[[2.0, 3.0]]], device=device)
    mu = torch.tensor([[2.0]], device=device)
    sigma = torch.tensor([[1e-7]], device=device)

    normalized = revin(x, mu, sigma)

    torch.testing.assert_close(
        normalized, normalized.new_tensor([[[0.0, 1.0]]])
    )


def test_revin_rejects_unsupported_shapes() -> None:
    x = torch.ones(2, 3, 4, 5)
    stats = torch.ones(2)

    with pytest.raises(ValueError, match="Unsupported shapes"):
        revin(x, stats, stats)


def test_revin_rejects_mismatched_stats() -> None:
    x = torch.ones(2, 3, 4)
    mu = torch.ones(2, 3)
    sigma = torch.ones(1, 3)

    with pytest.raises(
        ValueError, match="mu and sigma must have the same shape"
    ):
        revin(x, mu, sigma)


def test_revin_rejects_broadcastable_stats() -> None:
    x = torch.ones(2, 2, 3, 4)
    stats = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="leading dimensions"):
        revin(x, stats, stats)
