# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.processing import RandomProjection
from sdm.tensor import EnsembleTable


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.randn(6, 4),
        categorical=CategoricalTensor(
            torch.randint(0, 2, (6, 1)), categories=(torch.arange(2),)
        ),
    )


def test_random_projection() -> None:
    table = _table()

    inp = EnsembleTable(table, num_members=8)
    out = RandomProjection(8).fit_transform_ensemble(inp)

    assert out.num_groups == 1
    assert out.num_members == 8
    group = out._groups[0]
    assert group.size() == (8, 6, 9)
    assert group.numerical.size() == (8, 6, 8)
    assert group.numerical.stride() == (6 * 8, 8, 1)
    assert group.categorical.size() == (8, 6, 1)
    assert group.categorical.stride() == (0, 1, 1)
