import pytest
import torch
from sdm.models import TabICLv2
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("num_classes", [0, 10])
def test_tabiclv2(device: torch.device, num_classes: int) -> None:
    model = TabICLv2(
        num_classes=num_classes,
        num_quantiles=999,
        channels=16,
        num_embedding_layers=2,
        num_embedding_heads=2,
        num_inducing_points=8,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=2,
        num_icl_heads=4,
        device=device,
    )

    batch_size, num_rows, num_cols = 2, 8, 6
    num_train = 5

    x = torch.randn(batch_size, num_rows, num_cols, device=device)
    if num_classes > 0:
        y = torch.randint(
            0,
            num_classes,
            (batch_size, num_train),
            device=device,
        )
    else:
        y = torch.randn(batch_size, num_train, device=device)

    out = model(x, y)
    assert out.size() == (batch_size, num_rows - num_train, num_classes or 999)
    assert out.dtype == x.dtype
    assert out.device == x.device
