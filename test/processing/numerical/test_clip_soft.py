# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from sdm import TableTensor
from sdm.processing import ClipSoft
from sdm.testing import withCUDA


@withCUDA
def test_clip_soft_maps_finite_values(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-6.0, 0.0], [3.0, 6.0]],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(inp)

    out = ClipSoft(3.0).transform(table)

    expected = inp.new_tensor(
        [
            [-6 / math.sqrt(5), 0.0],
            [3 / math.sqrt(2), 6 / math.sqrt(5)],
        ],
    )
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_clip_soft_preserves_nan_and_maps_inf_to_bound(
    device: torch.device,
) -> None:
    bound = 5.0
    inp = torch.tensor(
        [[float("nan")], [float("inf")], [-float("inf")]],
        dtype=torch.float64,
        device=device,
    )

    out = ClipSoft(bound).transform(TableTensor.from_tensor(inp))

    expected = inp.new_tensor([[float("nan")], [bound], [-bound]])
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)


@withCUDA
def test_clip_soft_maps_overflow_to_bound(device: torch.device) -> None:
    inp = torch.tensor([[1e20], [-1e20]], dtype=torch.float32, device=device)

    out = ClipSoft(3.0).transform(TableTensor.from_tensor(inp))

    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[3.0], [-3.0]], device=device),
    )


@withCUDA
def test_clip_soft_preserves_leading_batch_dimensions(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[[-3.0], [0.0]], [[3.0], [6.0]]],
        dtype=torch.float64,
        device=device,
    )

    out = ClipSoft(3.0).transform(TableTensor.from_tensor(inp))

    expected = inp.new_tensor(
        [
            [[-3 / math.sqrt(2)], [0.0]],
            [[3 / math.sqrt(2)], [6 / math.sqrt(5)]],
        ],
    )
    torch.testing.assert_close(out.numerical, expected)
