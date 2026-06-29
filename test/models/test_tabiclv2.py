import pytest
import torch
from sdm.models import TabICLv2
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabiclv2(device: torch.device, dtype: torch.dtype) -> None:
    model = TabICLv2(pretrained=False, device=device)

    batch_size, num_rows, num_cols = 2, 8, 6
    num_train = 5

    x = torch.randn(batch_size, num_rows, num_cols, device=device)
    if torch.empty(0, dtype=dtype).is_floating_point():
        y = torch.randn(batch_size, num_train, device=device)
    else:
        y = torch.randint(0, 10, (batch_size, num_train), device=device)

    out = model(x, y)
    if y.is_floating_point():
        assert out.size() == (batch_size, num_rows - num_train, 999)
    else:
        assert out.size() == (batch_size, num_rows - num_train, 10)
    assert out.dtype == x.dtype
    assert out.device == x.device
