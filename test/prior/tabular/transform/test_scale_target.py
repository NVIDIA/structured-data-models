# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import ScaleTarget


def test_scale_target_scales_regression_target() -> None:
    x = TableTensor.from_tensor(torch.randn(50, 1))
    y = TableTensor.from_tensor(torch.randn(50, 1) * 5 + 3)
    labels = TableTensor(
        categorical=CategoricalTensor(
            code=torch.randint(3, (50, 1)), categories=(torch.arange(3),)
        )
    )

    _, scaled = ScaleTarget()(x=x, y=y)

    torch.testing.assert_close(scaled.numerical.mean(), torch.tensor(0.0))
    torch.testing.assert_close(scaled.numerical.std(), torch.tensor(1.0))
    assert ScaleTarget()(x=x, y=labels)[1] is labels
