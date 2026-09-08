import pytest
import torch

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "out_channels"),
    [(3, 4), (0, 5)],
)
@pytest.mark.parametrize(
    "num_key_value_heads_for_query",
    [
        pytest.param(None, id="mha"),
        pytest.param(1, id="mqa"),
    ],
)
def test_icl_block(
    device: torch.device,
    num_classes: int,
    out_channels: int,
    num_key_value_heads_for_query: int | None,
) -> None:
    block = ICLBlock(
        num_classes=num_classes,
        out_channels=out_channels,
        channels=8,
        num_layers=3,
        num_heads=2,
        device=device,
        num_key_value_heads_for_query=num_key_value_heads_for_query,
    )
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

    out = block(x.clone(), y)

    assert out.size() == (2, 2, out_channels)
    assert out.dtype == x.dtype
    assert out.device == device

    if num_key_value_heads_for_query is not None:
        mha = ICLBlock(
            num_classes=num_classes,
            out_channels=out_channels,
            channels=8,
            num_layers=3,
            num_heads=2,
            device=device,
        )
        mha.load_state_dict(block.state_dict())
        assert not torch.allclose(mha(x.clone(), y), out)

    cache = Cache()
    fit_out = block(x[..., :3, :].clone(), y, cache=cache)
    replayed = block(
        x[..., 3:, :].clone(),
        y[..., :0],
        cache=cache.freeze(),
    )

    assert fit_out.size() == (2, 0, out_channels)
    assert cache.size() > 0
    torch.testing.assert_close(replayed, out, atol=1e-4, rtol=1e-4)
