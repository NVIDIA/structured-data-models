# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models.timesfm3.recipe import default_recipe
from sdm.testing import withCUDA


@withCUDA
def test_output_recipe(device: torch.device) -> None:
    quantiles = torch.arange(9, dtype=torch.float32, device=device)
    first = torch.cat([100 + quantiles, quantiles.flip(-1)])
    second = torch.cat([100 + quantiles.flip(-1), 2 * quantiles])
    columns = [
        f"{target}__q{i}" for target in ("a", "b") for i in range(10, 100, 10)
    ]
    table = TableTensor.from_tensor(
        torch.stack([first, second]).unsqueeze(-2),
        columns=columns,
    )

    out = default_recipe().output.transform(table)

    expected = torch.cat([quantiles.new_full((9,), 104), 4 + quantiles / 2])
    torch.testing.assert_close(out.numerical, expected.unsqueeze(-2))
    assert out.columns == table.columns
