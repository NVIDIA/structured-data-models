# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import TableTensor
from sdm.models.kumo.timeseries.anomaly.preprocessing import AnomalyPreprocess
from sdm.models.kumo.timeseries.anomaly.recipe import default_recipe
from sdm.testing import withCUDA


@withCUDA
def test_scale_and_pad(device: torch.device) -> None:
    """Use reference ranges, retain constants, and append zero padding."""
    reference = TableTensor.from_tensor(
        torch.tensor([[1.0, 3.0], [5.0, 3.0]], device=device)
    )
    query = TableTensor.from_tensor(
        torch.tensor([[9.0, 4.0], [-3.0, 3.0]], device=device)
    )
    recipe = default_recipe(num_features=3, scale_factor=2)
    recipe.features.fit(reference)
    out = recipe.features.transform(query)
    expected = torch.tensor([[4.0, 2.0, 0.0], [-2.0, 0.0, 0.0]], device=device)
    torch.testing.assert_close(out.numerical, expected)


@withCUDA
def test_pca_and_state_dict(device: torch.device) -> None:
    """PCA and fitted ranges survive serialization and dtype moves."""
    # Orthogonal columns with distinct variances make component signs unique.
    x = torch.tensor(
        [
            [3.0, 2.0, 1.0],
            [3.0, -2.0, -1.0],
            [-3.0, 2.0, -1.0],
            [-3.0, -2.0, 1.0],
        ],
        device=device,
    )
    processor = AnomalyPreprocess(num_features=2)
    table = TableTensor.from_tensor(x)
    actual = processor.fit_transform(table)
    expected = (
        x[:, :2] + torch.tensor([3.0, 2.0], device=device)
    ) / torch.tensor([6.0, 4.0], device=device)
    torch.testing.assert_close(actual.numerical, expected)
    restored = AnomalyPreprocess(num_features=2).to(device)
    restored.load_state_dict(processor.state_dict())
    assert restored.is_fitted
    torch.testing.assert_close(restored(table).numerical, actual.numerical)
    restored = restored.double()
    torch.testing.assert_close(
        restored(table.double()).numerical, expected.double()
    )


def test_pca_reference_rows() -> None:
    """Do not silently reduce checkpoint width when PCA has too few rows."""
    table = TableTensor.from_tensor(torch.ones(2, 5))
    with pytest.raises(ValueError, match="reference rows"):
        AnomalyPreprocess(num_features=3).fit(table)
