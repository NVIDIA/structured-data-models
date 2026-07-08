import math

import pytest
import torch
from sdm.nn import RotaryEmbedding, apply_rotary_embedding
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


@withCUDA
def test_apply_interleaved_rotary_embedding(device: torch.device) -> None:
    frequencies = torch.full((2,), math.pi / 2, device=device)
    input = torch.zeros(4, 1, 4, device=device)
    input[..., 0] = 1
    input[..., 2] = 2

    output = apply_rotary_embedding(
        input,
        frequencies,
        layout="interleaved",
    )
    expected = torch.tensor(
        [
            [[1, 0, 2, 0]],
            [[0, 1, 0, 2]],
            [[-1, 0, -2, 0]],
            [[0, -1, 0, -2]],
        ],
        dtype=input.dtype,
        device=device,
    )

    torch.testing.assert_close(output, expected)

    with pytest.raises(ValueError, match="Unsupported rotary layout"):
        apply_rotary_embedding(
            input,
            frequencies,
            layout="invalid",  # ty: ignore[invalid-argument-type]
        )
