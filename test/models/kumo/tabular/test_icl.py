import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
def test_icl_block(device: torch.device) -> None:
    block = ICLBlock(
        num_classes=3,
        out_channels=4,
        channels=8,
        num_layers=3,
        num_heads=2,
        device=device,
    ).eval()
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(2, 5, 8, device=device)
    y = torch.randint(3, (2, 3), device=device)

    out = block(x, y)

    assert out.size() == (2, 2, 4)
    assert out.dtype == x.dtype
    assert out.device == device

    cache = Cache()
    fit_out = block(x[..., :3, :], y, cache=cache)
    replayed = block(x[..., 3:, :], y[..., :0], cache=cache.freeze())

    assert fit_out.size() == (2, 0, 4)
    assert cache.size() > 0
    torch.testing.assert_close(replayed, out, atol=1e-4, rtol=1e-4)
