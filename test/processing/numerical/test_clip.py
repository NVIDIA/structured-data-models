# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal

import pytest
import torch

from sdm import TableTensor
from sdm.processing import Clip
from sdm.testing import withCUDA


@withCUDA
def test_clip_clamps_fixed_bounds(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [[-101.0, -100.0], [100.0, 101.0]],
            device=device,
        ),
    )

    actual = Clip(-100.0, 100.0).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor(
            [[-100.0, -100.0], [100.0, 100.0]],
            device=device,
        ),
    )
    assert repr(Clip(-100.0, 100.0)) == "Clip(-100.0, 100.0)"


def test_clip_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="min_value"):
        Clip(1.0, -1.0)


@withCUDA
def test_clip_saturate_maps_finite_values(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-6.0, 0.0], [3.0, 6.0]],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(inp)

    out = Clip(-3.0, 3.0, method="saturate").transform(table)

    expected = inp.new_tensor(
        [
            [-6 / math.sqrt(5), 0.0],
            [3 / math.sqrt(2), 6 / math.sqrt(5)],
        ],
    )
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_clip_saturate_preserves_nan_and_maps_inf_to_bound(
    device: torch.device,
) -> None:
    bound = 5.0
    inp = torch.tensor(
        [[float("nan")], [float("inf")], [-float("inf")]],
        dtype=torch.float64,
        device=device,
    )

    out = Clip(-bound, bound, method="saturate").transform(
        TableTensor.from_tensor(inp)
    )

    expected = inp.new_tensor([[float("nan")], [bound], [-bound]])
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)


@withCUDA
def test_clip_saturate_maps_overflow_to_bound(device: torch.device) -> None:
    inp = torch.tensor([[1e20], [-1e20]], dtype=torch.float32, device=device)

    out = Clip(-3.0, 3.0, method="saturate").transform(
        TableTensor.from_tensor(inp)
    )

    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[3.0], [-3.0]], device=device),
    )


@pytest.mark.parametrize("method", ["hard", "saturate"])
def test_clip_inverse_leaves_clipped_values_unchanged(
    method: Literal["hard", "saturate"],
) -> None:
    table = TableTensor.from_tensor(torch.tensor([[-10.0], [1.0], [10.0]]))
    processor = Clip(-3.0, 3.0, method=method)
    clipped = processor.transform(table)

    torch.testing.assert_close(
        processor.inverse_transform(clipped).numerical,
        clipped.numerical,
    )
