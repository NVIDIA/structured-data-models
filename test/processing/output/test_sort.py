# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
def test_sort_quantiles_group(device: torch.device) -> None:
    data = torch.randn(3, 2, 3, device=device)
    table = TableTensor.from_tensor(
        data.flatten(-2),
        columns=["a__q10", "a__q50", "a__q90", "b__q10", "b__q50", "b__q90"],
    )

    out = SortQuantiles(num_quantiles=3).transform(table)
    assert out.size() == table.size()
    assert out.columns == table.columns
    assert out.device == table.device
    torch.testing.assert_close(
        out.numerical,
        data.sort(dim=-1).values.flatten(-2),
    )
