# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.nn import RevIN
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("affine_channels", [None, 2])
def test_revin(device: torch.device, affine_channels: int | None) -> None:
    norm = RevIN(affine_channels, device=device)
    x = torch.randn(2, 3, 2, 8, device=device)
    mean, std = norm.stats(x)
    torch.testing.assert_close(mean, x.mean(dim=-1, keepdim=True))
    torch.testing.assert_close(std, x.std(dim=-1, correction=1, keepdim=True))

    out = norm(x, mean, std)
    expected = (x - mean) / (std + norm.eps)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(norm.inverse(out, mean, std), x)


@withCUDA
def test_running_stats_update(device: torch.device) -> None:
    x = torch.tensor([[[1.0, 100.0, 2.0, 3.0]]], device=device)
    mask = torch.tensor([[[True, False, True, True]]], device=device)
    state = (
        torch.zeros(1, 1, 1, device=device, dtype=torch.int64),
        torch.zeros(1, 1, 1, device=device, dtype=torch.float64),
        torch.zeros(1, 1, 1, device=device, dtype=torch.float64),
    )

    count, mean, std = RevIN().running_stats(
        x=x,
        mask=mask,
        correction=0,
        state=state,
    )

    torch.testing.assert_close(count, count.new_tensor([[[3]]]))
    torch.testing.assert_close(mean, mean.new_tensor([[[2.0]]]))
    torch.testing.assert_close(std, std.new_tensor([[[2.0 / 3.0]]]).sqrt())


@withCUDA
def test_running_stats_fully_masked(device: torch.device) -> None:
    count = torch.tensor([[[3]]], device=device)
    mean = torch.tensor([[[2.0]]], device=device, dtype=torch.float64)
    std = torch.tensor([[[1.0]]], device=device, dtype=torch.float64)
    x = torch.tensor([[[100.0, 200.0]]], device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)

    actual = RevIN().running_stats(
        x=x,
        mask=mask,
        correction=0,
        state=(count, mean, std),
    )

    for output, expected in zip(actual, (count, mean, std)):
        torch.testing.assert_close(output, expected)


@withCUDA
def test_running_stats(device: torch.device) -> None:
    x = torch.tensor([[[[1.0, 3.0], [5.0, 99.0], [7.0, 9.0]]]], device=device)
    mask = torch.tensor(
        [[[[True, True], [True, False], [False, True]]]],
        device=device,
    )

    count, mean, std = RevIN().running_stats(x, mask, correction=0)

    torch.testing.assert_close(count, count.new_tensor([[[[2], [3], [4]]]]))
    torch.testing.assert_close(
        mean, mean.new_tensor([[[[2.0], [3.0], [4.5]]]])
    )
    torch.testing.assert_close(
        std,
        std.new_tensor([[[[1.0], [math.sqrt(8.0 / 3.0)], [math.sqrt(8.75)]]]]),
    )
