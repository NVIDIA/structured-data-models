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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_interleaved_rope(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    inv_freq = torch.tensor(
        [math.pi / 2, 0],
        device=device,
        dtype=dtype,
    )
    x = torch.tensor(
        [[[[1, 2, 3, 4]], [[1, 2, 3, 4]], [[1, 2, 3, 4]]]],
        device=device,
        dtype=dtype,
    )

    out = apply_rotary_embedding(
        x=x,
        inv_freq=inv_freq,
        layout="interleaved",
    )

    expected = torch.tensor(
        [[[[1, 2, 3, 4]], [[-2, 1, 3, 4]], [[-1, -2, 3, 4]]]],
        device=device,
        dtype=dtype,
    )

    torch.testing.assert_close(out, expected)
    assert out.dtype == x.dtype
    assert out.device == x.device


def test_apply_rotary_embedding_errors() -> None:
    x = torch.randn(2, 3, 1, 4)

    with pytest.raises(ValueError, match="Expected 2 channels, got 4"):
        apply_rotary_embedding(x=x, inv_freq=torch.ones(1))
