import math

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import PCA
from sdm.testing import withCUDA


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


@withCUDA
@pytest.mark.parametrize("shape", [(23, 7), (2, 3, 11, 7)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_components", [3, 50])
def test_fit_transform_matches_centered_projection(
    device: torch.device,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    num_components: int,
) -> None:
    # A large offset makes subtracting the mean after projection inaccurate.
    values = torch.randn(*shape, device=device, dtype=dtype).add_(2**20)
    table = TableTensor.from_tensor(values[..., ::2, :])
    before = table.numerical.clone()
    processor = PCA(num_components)

    actual = processor.fit_transform(table)
    expected = (before - processor.mean) @ processor.components
    torch.testing.assert_close(actual.numerical, expected)
    torch.testing.assert_close(table.numerical, before)
    torch.testing.assert_close(
        processor.transform(table).numerical, actual.numerical
    )

    restored = PCA(num_components).to(device=device, dtype=dtype)
    restored.load_state_dict(processor.state_dict())
    torch.testing.assert_close(
        restored.transform(table).numerical, actual.numerical
    )
