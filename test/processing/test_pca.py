import pytest
import torch
from sdm import Stype, TableTensor
from sdm.processing.text.pca import PCA


def _table(data: torch.Tensor) -> TableTensor:
    return TableTensor(
        columns={
            "numerical": tuple(f"x{i}" for i in range(data.size(-1))),
        },
        numerical=data,
    )


def test_pca_projects_to_requested_dim() -> None:
    torch.manual_seed(0)
    table = _table(torch.randn(20, 5))

    output = PCA(dim=2).fit_transform(table)

    assert output.numerical.size() == (20, 2)
    assert output.columns[Stype.numerical] == ("pca_0", "pca_1")


def test_pca_recovers_dominant_direction() -> None:
    # Rank-one data along (1, 1): the single component captures it exactly.
    steps = torch.arange(10, dtype=torch.get_default_dtype())
    table = _table(torch.stack((steps, steps), dim=-1))

    output = PCA(dim=1).fit_transform(table)

    centered_norm = (table.numerical - table.numerical.mean(dim=0)).norm(dim=1)
    assert torch.allclose(output.numerical.abs().squeeze(1), centered_norm)


def test_pca_caps_dim_at_data_rank() -> None:
    torch.manual_seed(0)
    table = _table(torch.randn(20, 3))

    output = PCA(dim=99).fit_transform(table)

    assert output.numerical.size() == (20, 3)


def test_pca_transform_uses_fitted_state() -> None:
    torch.manual_seed(0)
    pca = PCA(dim=2)
    pca.fit(_table(torch.randn(20, 4)))

    new = _table(torch.randn(5, 4))
    output1 = pca.transform(new)
    output2 = pca.transform(new)

    assert torch.equal(output1.numerical, output2.numerical)


def test_pca_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PCA(dim=2).transform(_table(torch.randn(4, 3)))


def test_pca_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        PCA(dim=0)
