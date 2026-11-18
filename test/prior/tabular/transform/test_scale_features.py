# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import ScaleFeatures


def test_scale_features_clips_and_standardizes_columns() -> None:
    numerical = torch.randn(50, 2) * 3 + 1
    numerical[0, 0] = 40.0
    code = torch.randint(4, (50, 1))
    x = TableTensor(
        columns={"numerical": ("x_0", "x_2"), "categorical": ("x_1",)},
        numerical=numerical,
        categorical=CategoricalTensor(
            code=code, categories=(torch.arange(4) * 10,)
        ),
    )
    y = TableTensor.from_tensor(torch.randn(50, 1))

    output, target = ScaleFeatures()(x=x, y=y)

    category = output.categorical.categories[0]
    values = torch.cat(
        (output.numerical, category[output.categorical.code]), dim=-1
    )
    torch.testing.assert_close(values.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(values.std(dim=0), torch.ones(3))
    assert values[0, 0] < 4.0
    assert output.categorical.code.equal(code)
    assert target is y
