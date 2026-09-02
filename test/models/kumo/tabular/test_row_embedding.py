import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_row_embedding(device: torch.device) -> None:
    encoder = RowEmbedding(
        num_classes=0,
        channels=16,
        num_layers=4,
        num_heads=2,
        num_inducing_points=4,
        num_readout_tokens=2,
        device=device,
    )
    K = encoder.readout_token.size(-2)
    features = torch.randn(2, 5, 3, 16, device=device)
    x = features.new_empty(2, 5, K + features.size(-2), 16)
    x[..., :, K:, :] = features
    y = torch.randn(2, 3, device=device)

    out = encoder(x, y)
    assert out.size() == (2, 5, 32)
    assert out.device == device

    cache = Cache()
    encoder(x[:, :3], y, cache=cache)
    out = encoder(x[:, 3:], y[:, :0], cache=cache.freeze())
    assert out.size() == (2, 2, 32)
    assert out.device == device
