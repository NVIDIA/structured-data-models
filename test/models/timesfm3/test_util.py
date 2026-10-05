# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

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
def test_revin(device: torch.device) -> None:
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device=device)
    mean = torch.tensor([[2.0]], device=device)
    std = torch.tensor([[0.5]], device=device)

    out = revin(revin(x, mean, std), mean, std, reverse=True)
    torch.testing.assert_close(out, x)


@withCUDA
def test_revin_near_zero_std(device: torch.device) -> None:
    x = torch.tensor([[[2.0, 3.0]]], device=device)
    mean = torch.tensor([[2.0]], device=device)
    std = torch.tensor([[1e-7]], device=device)

    out = revin(x, mean, std)
    torch.testing.assert_close(out, out.new_tensor([[[0.0, 1.0]]]))
