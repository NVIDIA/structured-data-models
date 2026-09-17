# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import ClipSoft
from sdm.testing import withCUDA


def _soft_clip(values: torch.Tensor, bound: float = 3.0) -> torch.Tensor:
    return values / (1 + (values / bound).square()).sqrt()


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

    out = ClipSoft().transform(table)

    torch.testing.assert_close(out.numerical, _soft_clip(inp))


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

    out = ClipSoft(max_absolute_value=bound).transform(
        TableTensor.from_tensor(inp)
    )

    expected = inp.new_tensor([[float("nan")], [bound], [-bound]])
    torch.testing.assert_close(out.numerical, expected, equal_nan=True)


@withCUDA
def test_clip_soft_maps_overflow_to_bound(device: torch.device) -> None:
    inp = torch.tensor([[1e20], [-1e20]], dtype=torch.float32, device=device)

    out = ClipSoft().transform(TableTensor.from_tensor(inp))

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

    out = ClipSoft().transform(TableTensor.from_tensor(inp))

    torch.testing.assert_close(out.numerical, _soft_clip(inp))


def test_clip_soft_rejects_invalid_bound() -> None:
    with pytest.raises(ValueError, match="max_absolute_value"):
        ClipSoft(max_absolute_value=0.0)
    with pytest.raises(ValueError, match="max_absolute_value"):
        ClipSoft(max_absolute_value=float("nan"))
    with pytest.raises(ValueError, match="max_absolute_value"):
        ClipSoft(max_absolute_value=float("inf"))
