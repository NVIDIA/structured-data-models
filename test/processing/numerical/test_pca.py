import torch

from sdm import Stype, TableTensor
from sdm.processing import PCA


def test_pca_projects_to_requested_num_components() -> None:
    table = TableTensor.from_tensor(torch.eye(20, 5, dtype=torch.float32))
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (20, 2)
    assert output.columns[Stype.numerical] == ("pca_0", "pca_1")


def test_pca_flattens_and_restores_leading_dimensions() -> None:
    data = torch.eye(24, 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    output = PCA(num_components=3).fit_transform(TableTensor.from_tensor(data))
    flat_output = PCA(num_components=3).fit_transform(
        TableTensor.from_tensor(data.reshape(-1, data.size(-1)))
    )
    assert output.numerical.size() == (2, 3, 4, 3)
    torch.testing.assert_close(
        output.numerical.reshape(-1, 3).abs(),
        flat_output.numerical.abs(),
    )


def test_pca_recovers_dominant_direction() -> None:
    # Rank-one data along (1, 1): the single component captures it exactly.
    steps = torch.arange(10, dtype=torch.float32)
    table = TableTensor.from_tensor(torch.stack((steps, steps), dim=-1))
    output = PCA(num_components=1).fit_transform(table)
    centered_norm = (table.numerical - table.numerical.mean(dim=0)).norm(dim=1)
    assert torch.allclose(output.numerical.abs().squeeze(1), centered_norm)


def test_pca_caps_num_components_at_feature_count() -> None:
    table = TableTensor.from_tensor(torch.eye(20, 3, dtype=torch.float32))
    output = PCA(num_components=99).fit_transform(table)
    assert output.numerical.size() == (20, 3)


def test_pca_caps_num_components_at_centered_rank() -> None:
    table = TableTensor.from_tensor(
        torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
        )
    )
    output = PCA(num_components=2).fit_transform(table)
    assert output.numerical.size() == (2, 1)
    assert output.columns[Stype.numerical] == ("pca_0",)


def test_pca_transform_uses_fitted_state() -> None:
    pca = PCA(num_components=3)
    fit_table = TableTensor.from_tensor(torch.eye(20, 4, dtype=torch.float32))
    pca.fit(fit_table)
    # Held-out rows shifted away from the fitted mean stay off-center;
    # recentering per call would zero the projection mean instead.
    held_out = TableTensor.from_tensor(
        torch.eye(20, 4, dtype=torch.float32)
        + torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=torch.float32)
    )
    assert pca.transform(fit_table).numerical.mean(dim=0).norm() < 1e-5
    assert pca.transform(held_out).numerical.mean(dim=0).norm() > 1e-2
