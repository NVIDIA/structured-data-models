import torch

from sdm.nn import SwiGLU
from sdm.testing import withCUDA


@withCUDA
def test_swiglu(device: torch.device) -> None:
    module = SwiGLU(channels=4, hidden_channels=7, device=device)

    out = module(torch.randn(2, 3, 4, device=device))
    assert out.size() == (2, 3, 4)
    assert out.device == device
