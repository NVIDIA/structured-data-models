import math

import pytest
import torch
from sdm.nn.rope import RotaryEmbedding


def test_rope() -> None:
    module = RotaryEmbedding(channels=2)
    with torch.no_grad():
        module.inv_freq.fill_(math.pi / 2)

    x = torch.zeros(4, 1, 2)
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
    )
    torch.testing.assert_close(out, expected)

    with pytest.raises(ValueError, match="Expected 2 channels, got 4"):
        module(torch.randn(2, 4, 3, 4))

    with pytest.raises(ValueError, match="`channels` must be even"):
        RotaryEmbedding(channels=3)
