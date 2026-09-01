import torch

from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_cell_embedding(device: torch.device) -> None:
    module = CellEmbedding(
        channels=8,
        group_size=3,
        num_frequencies=2,
        device=device,
    )

    x = torch.randn(6, 4, device=device)
    categorical_mask = torch.tensor([True, False, True, False], device=device)

    out = module(x, categorical_mask)
    assert out.size() == (6, 4, 8)
    assert out.device == device

    torch.testing.assert_close(
        module(x, categorical_mask, batch_size_limit=8),
        out,
    )
