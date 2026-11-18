# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior import Uniform
from sdm.prior.tabular.transform import MaskRandom


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.randn(100, 2),
        categorical=CategoricalTensor(
            code=torch.randint(3, (100, 2)),
            categories=(torch.arange(3),) * 2,
        ),
    )


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_mask_random_masks_cells_at_the_rate(rate: float) -> None:
    x = _table()
    y = TableTensor.from_tensor(torch.arange(100.0).reshape(-1, 1))

    masked, target = MaskRandom(rate=rate)(x, y)

    assert masked.numerical.isnan().eq(bool(rate)).all()
    assert masked.categorical.code.eq(-1).eq(bool(rate)).all()
    assert target.numerical.equal(y.numerical)


def test_mask_random_replays_with_the_same_generator() -> None:
    x = _table()
    y = TableTensor.from_tensor(torch.zeros(100, 1))
    transform = MaskRandom(rate=Uniform(0.2, 0.8))

    first, _ = transform(x, y, generator=torch.Generator().manual_seed(0))
    second, _ = transform(x, y, generator=torch.Generator().manual_seed(0))

    numerical = first.numerical.isnan()
    categorical = first.categorical.code == -1
    assert numerical.equal(second.numerical.isnan())
    assert categorical.equal(second.categorical.code == -1)
    assert first.numerical[~numerical].equal(x.numerical[~numerical])
    assert first.categorical.code[~categorical].equal(
        x.categorical.code[~categorical]
    )
