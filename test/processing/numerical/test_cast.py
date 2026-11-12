# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import Cast
from sdm.testing import onlyMPS, withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cast_converts_numerical_columns(
    device: torch.device, dtype: torch.dtype
) -> None:
    table = TableTensor(
        numerical=torch.tensor(
            [[1.5, float("nan")], [-2.0, 3.25]],
            device=device,
            dtype=dtype,
        ),
    )

    actual = Cast(torch.float64).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        table.numerical.double(),
        equal_nan=True,
    )


@withCUDA
def test_cast_mixed_member_dtypes(device: torch.device) -> None:
    first = TableTensor.from_tensor(
        torch.arange(6.0, device=device).view(3, 2)
    )
    second = first.replace_blocks(numerical=first.numerical.double())
    table = EnsembleTable.from_tables((first, second), (0, 1))

    actual = Cast(torch.float32).transform(table)

    for member in range(len(table)):
        torch.testing.assert_close(
            actual[member].numerical, table[member].numerical.float()
        )


@onlyMPS
def test_cast_mps_fallback() -> None:
    table = TableTensor(numerical=torch.randn(3, 1, device="mps"))
    actual = Cast(torch.float64).transform(table)
    assert actual.numerical.dtype == torch.float32
    torch.testing.assert_close(actual.numerical, table.numerical)
