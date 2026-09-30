# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import Cast
from sdm.testing import withCUDA


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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available",
)
def test_cast_float64_falls_back_to_float32_on_mps() -> None:
    table = TableTensor(
        numerical=torch.tensor(
            [[1.5, float("nan")], [-2.0, 3.25]],
            device="mps",
        ),
    )

    actual = Cast(torch.float64).transform(table)

    assert actual.numerical.dtype == torch.float32
    torch.testing.assert_close(
        actual.numerical,
        table.numerical,
        equal_nan=True,
    )
