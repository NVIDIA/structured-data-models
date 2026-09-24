# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import PCA


@pytest.mark.parametrize("shape", [(20, 5), (2, 3, 4, 4)])
def test_basic(shape: tuple[int, ...]) -> None:
    data = torch.eye(math.prod(shape[:-1]), shape[-1]).view(shape)
    table = TableTensor.from_tensor(data)
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (*shape[:-1], 2)
    assert output.columns[Stype.numerical] == ("pca_0", "pca_1")


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


def test_appends_components_and_holds_out_excluded_columns() -> None:
    generator = torch.Generator().manual_seed(0)
    numerical = torch.randn(8, 4, generator=generator)
    # A missing value in every column, which the projection must survive.
    numerical.diagonal().fill_(float("nan"))
    table = TableTensor(numerical=numerical)
    pca = PCA(2, append_original=True, exclude_columns=("num_0",))

    out = pca.fit_transform(table)
    components = out.numerical[:, 4:]

    assert out.columns[Stype.numerical] == (
        *table.columns[Stype.numerical],
        "pca_0",
        "pca_1",
    )
    torch.testing.assert_close(out.numerical[:, :4], numerical, equal_nan=True)
    assert components.isfinite().all()

    shifted = table.replace_blocks(numerical=numerical.clone())
    shifted.numerical[:, 0] += 100.0
    torch.testing.assert_close(
        pca.transform(shifted).numerical[:, 4:],
        components,
    )


def test_without_centering_the_projection_keeps_the_mean() -> None:
    table = TableTensor(numerical=torch.ones(4, 3))
    pca = PCA(1, center=False)

    out = pca.fit_transform(table)

    # A centered projection maps constant columns onto the origin.
    assert (pca.mean == 0).all()
    assert out.numerical.abs().min() > 0
