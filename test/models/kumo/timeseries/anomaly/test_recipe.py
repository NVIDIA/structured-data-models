# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.models.kumo.timeseries.anomaly import KumoAnomaly
from sdm.processing import PCA


@pytest.mark.parametrize("num_features", [3, 40, 45])
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_recipe_reduces_and_scales_features(
    num_features: int,
    batch_shape: tuple[int, ...],
) -> None:
    reference = TableTensor.from_tensor(
        torch.randn(*batch_shape, 50, num_features)
    )
    query = TableTensor.from_tensor(
        torch.randn(*batch_shape, 10, num_features)
    )
    recipe = KumoAnomaly.default_recipe()
    recipe.features.fit_ensemble(
        EnsembleTable.from_table(reference, num_members=1)
    )
    output = recipe.features.transform_ensemble(
        EnsembleTable.from_table(query, num_members=1)
    )[0]

    fitted = reference
    expected = query
    if num_features > 40:
        pca = PCA(num_components=40).fit(reference)
        fitted = pca.transform(reference)
        expected = pca.transform(query)

    minimum = fitted.numerical.amin(dim=-2, keepdim=True)
    scale = fitted.numerical.amax(dim=-2, keepdim=True) - minimum
    scale = scale.masked_fill(scale == 0, 1)
    assert output.columns == expected.columns
    torch.testing.assert_close(
        output.numerical, (expected.numerical - minimum) / scale
    )
