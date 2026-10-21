# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.processing import Cast
from sdm.testing import onlyMPS, withCUDA


@withCUDA
def test_cast_converts_numerical_columns(device: torch.device) -> None:
    table = TableTensor(
        numerical=torch.tensor(
            [[1.5, float("nan")], [-2.0, 3.25]],
            device=device,
        ),
    )

    actual = Cast(torch.float64).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        table.numerical.double(),
        equal_nan=True,
    )


@onlyMPS
def test_cast_mps_fallback() -> None:
    table = TableTensor(numerical=torch.randn(3, 1, device="mps"))
    actual = Cast(torch.float64).transform(table)
    assert actual.numerical.dtype == torch.float32
    torch.testing.assert_close(actual.numerical, table.numerical)
