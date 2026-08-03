import math

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import PCA


@pytest.mark.parametrize("shape", [(20, 5), (2, 3, 4, 4)])
def test_basic(
    shape: tuple[int, ...],
) -> None:
    data = torch.eye(
        math.prod(shape[:-1]),
        shape[-1],
        dtype=torch.float32,
    ).reshape(shape)
    table = TableTensor.from_tensor(data)
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (*shape[:-1], 2)
    assert output.columns[Stype.numerical] == ("pca_0", "pca_1")


def test_pca_projects_onto_fitted_dominant_direction() -> None:
    # Rank-one data along (1, 1): the single component captures it exactly.
    steps = torch.arange(10, dtype=torch.float32)
    fit_data = torch.stack((steps, steps), dim=-1)
    fit_table = TableTensor.from_tensor(fit_data)
    held_out = TableTensor.from_tensor(fit_data + 1)
    pca = PCA(num_components=1)
    fit_output = pca.fit_transform(fit_table)
    held_out_output = pca.transform(held_out)
    mean = fit_data.mean(dim=0)
    torch.testing.assert_close(
        fit_output.numerical.abs().squeeze(1),
        (fit_data - mean).norm(dim=1),
    )
    torch.testing.assert_close(
        held_out_output.numerical.abs().squeeze(1),
        (held_out.numerical - mean).norm(dim=1),
    )


def test_pca_caps_num_components_at_centered_rank() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (2, 1)
    assert output.columns[Stype.numerical] == ("pca_0",)
