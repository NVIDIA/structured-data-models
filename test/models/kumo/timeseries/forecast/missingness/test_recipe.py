# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models.kumo.timeseries.forecast.missingness.recipe import (
    StandardizeObserved,
)
from sdm.testing import withCUDA


@withCUDA
def test_observed_standardization(device: torch.device) -> None:
    x = torch.tensor(
        [
            [1.0, 5.0, torch.nan],
            [3.0, 5.0, torch.nan],
            [torch.nan, 5.0, torch.inf],
        ],
        device=device,
    )
    table = TableTensor.from_tensor(x)
    processor = StandardizeObserved()
    transformed = processor.fit_transform(table)
    expected = x.clone()
    expected[:2, 0] = x.new_tensor([-1.0, 1.0])
    expected[:, 1] = 0
    torch.testing.assert_close(transformed.numerical, expected, equal_nan=True)
    restored = processor.inverse_transform(transformed)
    torch.testing.assert_close(restored.numerical, x, equal_nan=True)
    query = TableTensor.from_tensor(x.new_tensor([[5.0, 6.0, 3.0]]))
    torch.testing.assert_close(
        processor.transform(query).numerical, x.new_tensor([[3.0, 1.0, 3.0]])
    )
