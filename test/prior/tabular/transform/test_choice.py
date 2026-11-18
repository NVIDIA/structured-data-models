# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import (
    Choice,
    Identity,
    MaskRandom,
    MaskRows,
)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_choice_applies_only_the_selected_transform(index: int) -> None:
    x = TableTensor.from_tensor(torch.ones(8, 1))
    y = TableTensor.from_tensor(torch.ones(8, 1))
    transform = Choice(
        Identity(),
        MaskRows(rows=(0.5, 0.5), columns=(1.0, 1.0)),
        MaskRandom(rate=1.0),
        weights=[float(i == index) for i in range(3)],
    )

    result = transform(x, y)

    assert result is not None
    features, target = result
    assert features.numerical.isnan().sum() == index * 4
    assert target is y
    assert x.numerical.isfinite().all()
