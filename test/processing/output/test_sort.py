# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.processing import SortQuantiles
from sdm.testing import withCUDA


@withCUDA
def test_sort_quantiles(device: torch.device) -> None:
    data = torch.randn(2, 3, device=device)
    table = TableTensor.from_tensor(data, columns=["q10", "q50", "q90"])

    out = SortQuantiles().transform(table)
    assert out.size() == table.size()
    assert out.columns == table.columns
    assert out.device == table.device
    torch.testing.assert_close(out.numerical, data.sort(dim=-1).values)


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_sort_quantiles_multiple_targets(
    device: torch.device,
    batch_shape: tuple[int, ...],
) -> None:
    data = torch.tensor([[30.0, 10.0, 20.0, 3.0, 1.0, 2.0]], device=device)
    data = data.expand(*batch_shape, 4, 6)
    table = TableTensor.from_tensor(
        data,
        columns=["a__q10", "a__q50", "a__q90", "b__q10", "b__q50", "b__q90"],
    )

    out = SortQuantiles(num_quantiles=3).transform(table)

    expected = torch.tensor([[10.0, 20.0, 30.0, 1.0, 2.0, 3.0]], device=device)
    torch.testing.assert_close(out.numerical, expected.expand_as(data))
    assert out.schema == table.schema
