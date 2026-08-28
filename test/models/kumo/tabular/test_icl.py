import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.nn import PerHeadLogNScale
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "out_channels"),
    [(3, 4), (0, 5)],
)
def test_icl_block(
    device: torch.device,
    num_classes: int,
    out_channels: int,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        out_channels=out_channels,
        channels=8,
        num_layers=3,
        num_heads=2,
        device=device,
    ).eval()
    for layer in block.layers:
        assert isinstance(layer, KumoTabularTransformerBlock)
        assert isinstance(layer.attn.sdpa.query_scaling, PerHeadLogNScale)
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    x = torch.randn(2, 5, 8, device=device)
    if num_classes == 0:
        y = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.75]],
            device=device,
        )
    else:
        y = torch.randint(num_classes, (2, 3), device=device)

    out = block(x, y)

    assert out.size() == (2, 2, out_channels)
    assert out.dtype == x.dtype
    assert out.device == device

    cache = Cache()
    fit_out = block(x[..., :3, :], y, cache=cache)
    replayed = block(x[..., 3:, :], y[..., :0], cache=cache.freeze())

    assert fit_out.size() == (2, 0, out_channels)
    assert cache.size() > 0
    torch.testing.assert_close(replayed, out, atol=1e-4, rtol=1e-4)
