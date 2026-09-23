# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.processing import Cast
from sdm.testing import withCUDA


@withCUDA
def test_cast_converts_numerical_columns(device: torch.device) -> None:
    table = TableTensor(
        numerical=torch.tensor(
            [[1.5, float("nan")], [-2.0, 3.25]],
            device=device,
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], device=device),
            categories=(torch.arange(2, device=device),),
        ),
    )

    actual = Cast(torch.float64).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        table.numerical.double(),
        equal_nan=True,
    )
    assert actual.categorical.equal(table.categorical)
    assert repr(Cast(torch.float64)) == "Cast(torch.float64)"


def test_cast_rejects_non_floating_point_dtype() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        Cast(torch.int64)
