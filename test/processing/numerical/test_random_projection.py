# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, EnsembleTable, TableTensor
from sdm.processing import RandomProjection


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.randn(6, 4),
        categorical=CategoricalTensor(
            torch.randint(0, 2, (6, 1)), categories=(torch.arange(2),)
        ),
    )


def test_random_projection() -> None:
    table = _table()

    inp = EnsembleTable.from_table(table, num_members=8)
    out = RandomProjection(8).fit_transform(inp)

    assert len(out) == 8
    group = out.as_tensor()
    assert group.size() == (8, 6, 9)
    assert group.numerical.size() == (8, 6, 8)
    assert group.numerical.stride() == (6 * 8, 8, 1)
    assert group.categorical.size() == (8, 6, 1)
    for member in range(len(out)):
        torch.testing.assert_close(
            out[member].categorical.code, table.categorical.code
        )
