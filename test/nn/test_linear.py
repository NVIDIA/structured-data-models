import pytest
import torch

from sdm.nn.linear import Linear
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("shape", [(2, 0, 3), (2, 4, 0, 3), (0, 4, 3), (2, 4, 3)])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_out(
    device: torch.device, shape: tuple[int, ...], bias: bool
) -> None:
    layer = Linear(3, 5, bias=bias, device=device)
    inputs = torch.randn(shape, device=device)
    output = torch.empty((*shape[:-1], 5), device=device)

    with torch.no_grad():
        expected = layer(inputs)
        actual = layer(inputs, out=output)

    assert actual is output
    torch.testing.assert_close(actual, expected)
