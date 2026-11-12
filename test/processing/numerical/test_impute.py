# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import ImputeConstant, ImputeMean
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_impute_mean(device: torch.device, dtype: torch.dtype | None) -> None:
    inp = torch.tensor(
        [
            [
                [1.0, torch.nan, torch.nan],
                [3.0, 5.0, torch.nan],
                [torch.nan, 7.0, torch.nan],
            ],
            [
                [10.0, 2.0, torch.nan],
                [20.0, torch.nan, torch.nan],
                [30.0, 6.0, torch.nan],
            ],
        ],
        dtype=dtype,
        device=device,
    )

    fill_value = -5.0
    processor = ImputeMean(fill_value=fill_value).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(
        transformed,
        torch.tensor(
            [
                [
                    [1.0, 6.0, fill_value],
                    [3.0, 5.0, fill_value],
                    [2.0, 7.0, fill_value],
                ],
                [
                    [10.0, 2.0, fill_value],
                    [20.0, 4.0, fill_value],
                    [30.0, 6.0, fill_value],
                ],
            ],
            dtype=dtype,
            device=device,
        ),
    )


@withCUDA
def test_impute_mean_infinite(device: torch.device) -> None:
    inp = torch.tensor(
        [
            [1.0, torch.inf, torch.inf],
            [3.0, 2.0, -torch.inf],
            [torch.nan, torch.nan, torch.nan],
            [-torch.inf, 4.0, torch.nan],
        ],
        device=device,
    )

    processor = ImputeMean(fill_value=-5.0).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    torch.testing.assert_close(
        transformed,
        torch.tensor(
            [
                [1.0, torch.inf, torch.inf],
                [3.0, 2.0, -torch.inf],
                [2.0, 3.0, -5.0],
                [-torch.inf, 4.0, -5.0],
            ],
            device=device,
        ),
    )


@withCUDA
@pytest.mark.parametrize("fill_value", [0.0, -5.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_impute_constant(
    device: torch.device,
    fill_value: float,
    dtype: torch.dtype,
) -> None:
    inp = torch.tensor(
        [
            [[torch.nan, 2.0], [torch.inf, -torch.inf]],
            [[3.0, torch.nan], [torch.nan, torch.nan]],
        ],
        device=device,
        dtype=dtype,
    )
    table = TableTensor.from_tensor(inp)
    processor = (
        ImputeConstant()
        if fill_value == 0.0
        else ImputeConstant(fill_value=fill_value)
    )
    out = processor.transform(table)
    expected = torch.tensor(
        [
            [[fill_value, 2.0], [torch.inf, -torch.inf]],
            [[3.0, fill_value], [fill_value, fill_value]],
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(out.numerical, expected)
    torch.testing.assert_close(table.numerical, inp, equal_nan=True)
    torch.testing.assert_close(
        processor.inverse_transform(out).numerical, expected
    )
