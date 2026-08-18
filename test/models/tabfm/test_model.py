import pytest
import torch

from sdm.cache import Cache
from sdm.models.tabfm.model import _TabFM
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_forward(device: torch.device, dtype: torch.dtype) -> None:
    model = _TabFM(
        num_classes=0 if dtype.is_floating_point else 10,
        channels=64,
        num_inducing_points=128,
        num_readout_tokens=4,
        num_icl_layers=4,
        device=device,
    )

    x = torch.randn(8, 6, device=device)
    categorical_mask = torch.tensor([True, False] * 3, device=device)
    if dtype.is_floating_point:
        y = torch.randn(5, device=device)
    else:
        y = torch.randint(0, 10, (5,), device=device)

    out1 = model(x, y, categorical_mask)
    assert out1.dtype == x.dtype
    assert out1.device == device
    if dtype.is_floating_point:
        assert out1.size() == (3, 1)
    else:
        assert out1.size() == (3, 10)

    cache = Cache()
    model(x[: y.size(0)], y, categorical_mask, cache=cache)
    cache.freeze()
    out2 = model(x[y.size(0) :], y[:0], categorical_mask, cache=cache)
    torch.testing.assert_close(out1, out2)
