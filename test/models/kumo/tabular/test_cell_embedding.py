import torch

from sdm.models.kumo.tabular.cell_embedding import (
    FourierNanIndicatorCellEmbedding,
)
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_fourier_nan_indicator_cell_embedding(device: torch.device) -> None:
    module = FourierNanIndicatorCellEmbedding(
        channels=8,
        group_size=3,
        num_frequencies=2,
        device=device,
    )
    x = torch.randn(2, 6, 4, device=device)
    x[0, 0, 0] = torch.nan
    x[0, 1, 1] = torch.nan
    x[0, 4, 2] = torch.nan
    x[1, :3, 3] = torch.nan
    categorical_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]],
        device=device,
    )

    out = module(x, categorical_mask, train_size=3)
    assert out.size() == (2, 6, 4, 8)
    assert out.device == device
    assert out.isfinite().all()
    torch.testing.assert_close(
        module(
            x,
            categorical_mask,
            train_size=3,
            batch_size_limit=8,
        ),
        out,
    )


def test_fourier_nan_indicator_imputes_from_context_only() -> None:
    module = FourierNanIndicatorCellEmbedding(
        channels=4,
        group_size=3,
        num_frequencies=2,
    )
    with torch.no_grad():
        module.nan_lin.weight.zero_()
    categorical_mask = torch.tensor([False])

    x = torch.tensor([1.0, 3.0, torch.nan, 100.0]).view(4, 1)
    expected = torch.tensor([1.0, 3.0, 2.0, 100.0]).view(4, 1)
    torch.testing.assert_close(
        module(x, categorical_mask, train_size=2),
        module(expected, categorical_mask, train_size=2),
    )

    x = torch.tensor([torch.nan, torch.nan, torch.nan, 7.0]).view(4, 1)
    expected = torch.tensor([0.0, 0.0, 0.0, 7.0]).view(4, 1)
    torch.testing.assert_close(
        module(x, categorical_mask, train_size=2),
        module(expected, categorical_mask, train_size=2),
    )


def test_fourier_nan_indicator_matches_finite_inputs() -> None:
    fourier = CellEmbedding(
        channels=4,
        group_size=3,
        num_frequencies=2,
    )
    module = FourierNanIndicatorCellEmbedding(
        channels=4,
        group_size=3,
        num_frequencies=2,
    )
    module.load_state_dict(fourier.state_dict(), strict=False)
    x = torch.randn(2, 5, 4)
    categorical_mask = torch.tensor(
        [[True, False, True, False], [False, True, False, True]]
    )

    torch.testing.assert_close(
        module(x, categorical_mask, train_size=3),
        fourier(x, categorical_mask),
    )


def test_fourier_nan_indicator_is_additive() -> None:
    module = FourierNanIndicatorCellEmbedding(
        channels=4,
        group_size=3,
        num_frequencies=2,
    )
    with torch.no_grad():
        module.nan_lin.weight.fill_(1.0)
    categorical_mask = torch.tensor([False])
    missing = torch.tensor([1.0, 3.0, torch.nan]).view(3, 1)
    observed = torch.tensor([1.0, 3.0, 2.0]).view(3, 1)

    out = module(missing, categorical_mask, train_size=2)
    expected = module(observed, categorical_mask, train_size=2)
    torch.testing.assert_close(out[-1] - expected[-1], torch.full((1, 4), 3.0))
