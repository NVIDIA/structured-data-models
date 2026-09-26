# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.models import KumoForecasting
from sdm.models.kumo.timeseries.forecasting.normalization import RevIN
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


def test_fixed_standardizer_affects_revin_epsilon() -> None:
    # Use float64 to isolate epsilon scaling from float32 centering roundoff.
    values = torch.tensor(
        [[-2e-5], [-1e-5], [1e-5], [2e-5]], dtype=torch.float64
    )
    table = TableTensor.from_tensor(values, columns=["a"])
    recipe = KumoForecasting.default_recipe()
    standardized = recipe.features.fit_transform(table).numerical.T.unsqueeze(
        0
    )
    raw = values.T.unsqueeze(0)
    revin = RevIN(num_features=1)
    actual, _ = revin(standardized)
    without_standardizer, _ = revin(raw)
    # The released standardizer makes epsilon scale-dependent in raw units.
    expected = (raw - raw.mean(dim=-1, keepdim=True)) / (
        raw.std(dim=-1, correction=0, keepdim=True)
        + 8.693312644958496 * revin.eps
    )
    torch.testing.assert_close(actual, expected)
    assert (actual - without_standardizer).abs().max() > 0.1
