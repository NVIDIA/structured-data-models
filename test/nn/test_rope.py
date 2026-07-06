import math

import pytest
import torch
from sdm.nn import RotaryEmbedding
from sdm.testing import withCUDA


@withCUDA
def test_rope(device: torch.device) -> None:
    module = RotaryEmbedding(channels=2, device=device)
    with torch.no_grad():
        module.inv_freq.fill_(math.pi / 2)

    x = torch.zeros(4, 1, 2, device=device)
    x[..., 0] = 1

    out = module(x)
    expected = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
            [[-1, 0]],
            [[0, -1]],
        ],
        dtype=x.dtype,
        device=device,
    )
    torch.testing.assert_close(out, expected)

    with pytest.raises(ValueError, match="Expected 2 channels, got 4"):
        module(torch.randn(2, 4, 3, 4, device=device))

    with pytest.raises(ValueError, match="`channels` must be even"):
        RotaryEmbedding(channels=3, device=device)


def test_rope_inverse_frequencies_stay_float32() -> None:
    module = RotaryEmbedding(channels=4)
    module.to(torch.bfloat16)
    # Rotary phases are precision-critical: casting the module must not
    # round the inverse frequencies - the exact float32 values must
    # survive, not merely the dtype.
    assert module.inv_freq.dtype == torch.float32
    torch.testing.assert_close(
        module.inv_freq,
        RotaryEmbedding(channels=4).inv_freq,
        atol=0.0,
        rtol=0.0,
    )

    x = torch.randn(2, 5, 3, 4, dtype=torch.bfloat16)
    out = module(x)
    assert out.dtype == torch.bfloat16

    reference = RotaryEmbedding(channels=4)
    expected = reference(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)
