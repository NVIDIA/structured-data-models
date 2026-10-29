# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.timesfm3.util import (
    crossfade_patches,
    gather_future_patches,
)
from sdm.testing import withCUDA


@withCUDA
def test_gather_future_patches(device: torch.device) -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        device=device,
    ).view(1, 1, 4, 2)

    expected_out = x.new_tensor(
        [
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    ).view(1, 1, 4, 4)
    expected_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, True, True],
            [True, True, True, True],
        ],
        device=device,
    ).view(1, 1, 4, 4)

    out, mask = gather_future_patches(x, num_patches=2)

    torch.testing.assert_close(out, expected_out)
    assert torch.equal(mask, expected_mask)


@withCUDA
def test_crossfade_patches(device: torch.device) -> None:
    x = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        ],
        device=device,
    ).view(1, 1, 2, 6, 1)

    expected = x.new_tensor(
        [0.0, 1.0, 2.0, 3.0, 7.5, 12.0, 13.0, 14.0, 15.0]
    ).view(1, 1, 9, 1)

    output = crossfade_patches(x, step=3, dim=2)

    torch.testing.assert_close(output, expected)
