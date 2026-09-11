import pytest
import torch

from sdm.nn.linear import Linear
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "shape", [(2, 0, 3), (2, 4, 0, 3), (0, 4, 3), (2, 4, 3)]
)
def test_linear(device: torch.device, shape: tuple[int, ...]) -> None:
    layer = Linear(3, 5, device=device)
    x = torch.randn(shape, device=device)
    out = torch.empty((*shape[:-1], 5), device=device)

    expected = layer(x)
    with torch.no_grad():
        actual = layer(x, out=out)

    assert actual is out
    torch.testing.assert_close(actual, expected)
