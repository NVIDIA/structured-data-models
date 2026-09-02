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

    prefixed = module(x, categorical_mask, num_prefix_columns=2)
    assert prefixed.size() == (6, 6, 8)
    torch.testing.assert_close(prefixed[..., 2:, :], out)

    categorical_mask = torch.zeros(4, dtype=torch.bool, device=device)
    out = module(x, categorical_mask)
    fast_out = module(
        x,
        categorical_mask,
        num_categorical_columns=0,
    )
    torch.testing.assert_close(fast_out, out)

    categorical_mask = torch.ones(4, dtype=torch.bool, device=device)
    out = module(x, categorical_mask)
    fast_out = module(
        x,
        categorical_mask,
        num_categorical_columns=4,
    )
    torch.testing.assert_close(fast_out, out)

    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.amp.autocast(device.type, dtype=dtype):
        out = module(
            x,
            torch.zeros(4, dtype=torch.bool, device=device),
            num_categorical_columns=0,
        )
    assert out.dtype == dtype
