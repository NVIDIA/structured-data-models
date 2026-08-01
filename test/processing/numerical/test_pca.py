import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import PCA


def _table(data: torch.Tensor) -> TableTensor:
    return TableTensor(
        columns={
            "numerical": tuple(f"x{i}" for i in range(data.size(-1))),
        },
        numerical=data,
    )


def _full_rank_data(num_rows: int, num_columns: int) -> torch.Tensor:
    steps = torch.linspace(
        -1,
        1,
        steps=num_rows,
        dtype=torch.get_default_dtype(),
    )
    return torch.stack(
        [steps.pow(i + 1) for i in range(num_columns)],
        dim=-1,
    )


def test_pca_projects_to_requested_num_components() -> None:
    table = _table(_full_rank_data(20, 5))

    output = PCA(num_components=2).fit_transform(table)

    assert output.numerical.size() == (20, 2)
    assert output.columns[Stype.numerical] == ("pca_0", "pca_1")


def test_pca_flattens_and_restores_leading_dimensions() -> None:
    data = _full_rank_data(24, 4).reshape(2, 3, 4, 4)

    output = PCA(num_components=3).fit_transform(_table(data))
    flat_output = PCA(num_components=3).fit_transform(
        _table(data.reshape(-1, data.size(-1)))
    )

    assert output.numerical.size() == (2, 3, 4, 3)
    torch.testing.assert_close(
        output.numerical.reshape(-1, 3).abs(),
        flat_output.numerical.abs(),
    )


def test_pca_recovers_dominant_direction() -> None:
    # Rank-one data along (1, 1): the single component captures it exactly.
    steps = torch.arange(10, dtype=torch.get_default_dtype())
    table = _table(torch.stack((steps, steps), dim=-1))

    output = PCA(num_components=1).fit_transform(table)

    centered_norm = (table.numerical - table.numerical.mean(dim=0)).norm(dim=1)
    assert torch.allclose(output.numerical.abs().squeeze(1), centered_norm)


def test_pca_caps_num_components_at_feature_count() -> None:
    table = _table(_full_rank_data(20, 3))

    output = PCA(num_components=99).fit_transform(table)

    assert output.numerical.size() == (20, 3)


def test_pca_caps_num_components_at_centered_rank() -> None:
    table = _table(
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
    pca = PCA(num_components=2)
    fit_table = _table(_full_rank_data(20, 4))
    pca.fit(fit_table)

    # Held-out rows shifted away from the fitted mean stay off-center;
    # recentering per call would zero the projection mean instead.
    held_out = _table(_full_rank_data(20, 4) + 1.0)

    assert pca.transform(fit_table).numerical.mean(dim=0).abs().max() < 1e-5
    assert pca.transform(held_out).numerical.mean(dim=0).abs().min() > 1e-2


def test_pca_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PCA(num_components=2).transform(
            _table(
                torch.arange(12, dtype=torch.get_default_dtype()).view(4, 3)
            )
        )


def test_pca_rejects_non_positive_num_components() -> None:
    with pytest.raises(ValueError, match="positive"):
        PCA(num_components=0)
