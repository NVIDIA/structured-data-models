# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models import KumoForecasting
from sdm.processing import InvertibleMixin


def test_fixed_standardizer_and_inverse() -> None:
    recipe = KumoForecasting.default_recipe()
    values = torch.tensor([[1.0, 20.0], [3.0, 40.0], [5.0, 60.0]])
    table = TableTensor.from_tensor(values, columns=["a", "b"])
    transformed = recipe.target.fit_transform(table)
    assert isinstance(recipe.target, InvertibleMixin)
    torch.testing.assert_close(
        transformed.numerical, (values + 5.671202659606934) / 8.693312644958496
    )
    torch.testing.assert_close(
        recipe.target.inverse_transform(transformed).numerical, values
    )
    assert transformed.columns == table.columns
    features = recipe.features.fit_transform(table)
    torch.testing.assert_close(features.numerical, transformed.numerical)
    # The transform is independent of the supplied context statistics.
    recipe.features.fit(table.replace_blocks(numerical=values * 100))
    torch.testing.assert_close(
        recipe.features.transform(table).numerical, features.numerical
    )
