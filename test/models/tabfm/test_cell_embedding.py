import pytest
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
    ).eval()

    x = torch.randn(6, 4, device=device)
    categorical_mask = torch.tensor([True, False, True, False], device=device)

    with torch.no_grad():
        out1 = module(x, categorical_mask)
    assert out1.size() == (6, 4, 8)
    assert out1.device == device

    with torch.no_grad():
        out2 = module(x, categorical_mask, batch_size_limit=8)
    torch.testing.assert_close(out1, out2)

    buffer = torch.empty(6, 4, 8, device=device)
    with torch.no_grad():
        out3 = module(x, categorical_mask, out=buffer)
    assert out3 is buffer
    torch.testing.assert_close(out3, out1)

    with pytest.raises(RuntimeError, match="only supported when gradients"):
        module(x, categorical_mask, out=buffer)

    out4 = module(x, categorical_mask)
    torch.testing.assert_close(out4, out1)
    out4.sum().backward()
    assert module.num_lin.weight.grad is not None
    assert module.cat_lin.weight.grad is not None
