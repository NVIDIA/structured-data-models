# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm import TableTensor
from sdm.processing import PCA


@pytest.mark.parametrize("shape", [(20, 5), (2, 3, 4, 4)])
def test_basic(shape: tuple[int, ...]) -> None:
    data = torch.eye(math.prod(shape[:-1]), shape[-1]).view(shape)
    table = TableTensor.from_tensor(data)
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (*shape[:-1], 2)
    assert output.numerical_columns == ("pca_0", "pca_1")


@pytest.mark.parametrize("num_components", [3, 5])
@pytest.mark.parametrize("shape", [(2, 3), (2, 4, 3)])
def test_pca_preserves_features_without_reduction(
    num_components: int,
    shape: tuple[int, ...],
) -> None:
    table = TableTensor.from_tensor(torch.randn(shape))
    pca = PCA(num_components=num_components)
    output = pca.fit_transform(table)
    assert output.columns == table.columns
    torch.testing.assert_close(output.numerical, table.numerical)

    query = TableTensor.from_tensor(torch.randn(shape))
    output = pca.transform(query)
    assert output.columns == query.columns
    torch.testing.assert_close(output.numerical, query.numerical)


def test_pca_projects_onto_fitted_dominant_direction() -> None:
    steps = torch.arange(10, dtype=torch.float)
    fit_data = torch.stack((steps, steps), dim=-1)
    fit_mean = fit_data.mean(dim=-2, keepdim=True)
    fit_table = TableTensor.from_tensor(fit_data)
    pca = PCA(num_components=1)

    out = pca.fit_transform(fit_table)
    torch.testing.assert_close(
        out.numerical.abs().squeeze(1),
        (fit_data - fit_mean).norm(dim=1),
    )

    transform_table = TableTensor.from_tensor(fit_data + 1)
    out = pca.transform(transform_table)
    torch.testing.assert_close(
        out.numerical.abs().squeeze(1),
        (transform_table.numerical - fit_mean).norm(dim=1),
    )
