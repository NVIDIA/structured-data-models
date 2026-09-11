import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.cell_embedding import CellEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_cell_embedding(device: torch.device) -> None:
    module = CellEmbedding(
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

    expected = module(x, categorical_mask, train_size=3)
    with torch.no_grad():
        out = module(x, categorical_mask, train_size=3, cache=Cache())
        buffer = out.new_empty((2, 6, 5, 8))[..., 1:, :]
        chunked = module(
            x=x,
            categorical_mask=categorical_mask,
            train_size=3,
            batch_size_limit=8,
            out=buffer,
        )
    assert out.size() == (2, 6, 4, 8)
    assert out.device == device
    assert out.isfinite().all()
    assert chunked is buffer
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(chunked, expected)


def test_cell_embedding_imputes_from_context_only() -> None:
    module = CellEmbedding(
        channels=4,
        group_size=1,
        num_frequencies=2,
    )
    with torch.no_grad():
        module.nan_lin.weight.fill_(1.0)
    categorical_mask = torch.tensor([False, False])
    x = torch.tensor(
        [
            [1.0, torch.nan],
            [3.0, torch.nan],
            [torch.nan, torch.nan],
            [100.0, 7.0],
        ],
    )
    imputed = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [100.0, 7.0]])

    out = module(x, categorical_mask, train_size=2)
    expected = module(imputed, categorical_mask, train_size=2)
    torch.testing.assert_close(out, expected + x.isnan().unsqueeze(-1))
    torch.testing.assert_close(
        module(
            x=x,
            categorical_mask=categorical_mask,
            impute_mean=torch.tensor([[2.0, 0.0]]),
        ),
        out,
    )
