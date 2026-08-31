import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.table_encoder import TableEncoder
from sdm.testing import withCUDA


@withCUDA
def test_table_encoder(device: torch.device) -> None:
    encoder = TableEncoder(
        channels=16,
        num_layers=4,
        num_col_heads=2,
        num_row_heads=2,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    )
    context = torch.randn(2, 3, 3, 16, device=device)
    query = torch.randn(2, 2, 3, 16, device=device)

    out = encoder(
        torch.cat((context, query), dim=-3),
        num_context_rows=context.size(-3),
    )
    assert out.size() == (2, 5, 32)
    assert out.device == device

    cache = Cache()
    encoder(context, num_context_rows=context.size(-3), cache=cache)
    out = encoder(query, num_context_rows=0, cache=cache.freeze())
    assert out.size() == (2, 2, 32)
    assert out.device == device

    with torch.no_grad():
        encoder.norm.weight.zero_()
    out = encoder(
        torch.cat((context, query), dim=-3),
        num_context_rows=context.size(-3),
    )
    assert torch.count_nonzero(out) == 0
