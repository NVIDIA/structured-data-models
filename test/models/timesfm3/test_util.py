# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.models.timesfm3.util import (
    crossfade_patches,
    gather_future_patches,
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


@withCUDA
def test_gather_future_patches(device: torch.device) -> None:
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device=device
    ).reshape(1, 1, 4, 2)

    expected = x.new_tensor(
        [
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ).reshape(1, 1, 4, 4)
    expected_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, True, True],
            [True, True, True, True],
        ],
        device=device,
    ).reshape(1, 1, 4, 4)

    output, wrap_mask = gather_future_patches(x, num_future_patches=2)

    torch.testing.assert_close(output, expected)
    assert torch.equal(wrap_mask, expected_mask)


@pytest.mark.parametrize(
    ("num_patches", "num_future_patches"),
    [(1, 2), (2, 3), (5, 6)],
)
@withCUDA
def test_gather_future_patches_varied_sizes(
    device: torch.device,
    num_patches: int,
    num_future_patches: int,
) -> None:
    patch_len = 2
    x = torch.arange(num_patches * patch_len, device=device).reshape(
        1, 1, num_patches, patch_len
    )

    output, wrap_mask = gather_future_patches(x, num_future_patches)

    expected = torch.stack(
        [
            torch.cat(
                [
                    x[..., (patch + offset) % num_patches, :]
                    for offset in range(1, num_future_patches + 1)
                ],
                dim=-1,
            )
            for patch in range(num_patches)
        ],
        dim=-2,
    )
    expected_mask = torch.tensor(
        [
            [
                patch + offset >= num_patches
                for offset in range(1, num_future_patches + 1)
            ]
            for patch in range(num_patches)
        ],
        device=device,
    ).repeat_interleave(patch_len, dim=-1)

    torch.testing.assert_close(output, expected)
    assert torch.equal(wrap_mask, expected_mask[None, None])


@withCUDA
def test_crossfade_patches(device: torch.device) -> None:
    patch_preds = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
        ],
        device=device,
        dtype=torch.float64,
    ).reshape(1, 1, 2, 6, 1)

    expected = patch_preds.new_tensor(
        [[0.0], [1.0], [2.0], [3.0], [7.5], [12.0], [13.0], [14.0], [15.0]]
    ).reshape(1, 1, 9, 1)

    output = crossfade_patches(patch_preds, step=3)

    assert output.dtype == patch_preds.dtype
    assert output.device == device
    torch.testing.assert_close(output, expected)


@withCUDA
def test_crossfade_single_patch(device: torch.device) -> None:
    patch_preds = torch.arange(6, device=device).reshape(1, 1, 1, 3, 2)

    output = crossfade_patches(patch_preds, step=2)

    assert output.dtype == patch_preds.dtype
    assert torch.equal(output, patch_preds[:, :, 0])
