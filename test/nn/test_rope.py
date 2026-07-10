import math
from typing import Literal

import pytest
import torch
from sdm.nn import RotaryEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
def test_rope(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
) -> None:
    if layout == "split_half":
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[-3.0, 2.0, 1.0, 4.0]],
                [[-1.0, 2.0, -3.0, 4.0]],
            ],
            device=device,
        )
    else:
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[-2.0, 1.0, 3.0, 4.0]],
                [[-1.0, -2.0, 3.0, 4.0]],
            ],
            device=device,
        )

    module = RotaryEmbedding(channels=4, layout=layout, device=device)

    with torch.no_grad():
        inv_freq = torch.tensor([math.pi / 2, 0], device=device)
        module.inv_freq.copy_(inv_freq)

    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0, 4.0]],
        ],
        device=device,
    )

    out = module(x)
    torch.testing.assert_close(out, expected)

    with pytest.raises(ValueError, match="Expected 4 channels"):
        module(torch.randn(2, 4, 3, 2, device=device))

    with pytest.raises(ValueError, match="'channels' must be even"):
        RotaryEmbedding(channels=3, layout=layout, device=device)


@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
def test_rope_inverse_frequencies_stay_float32(
    layout: Literal["split_half", "interleaved"],
) -> None:
    torch.manual_seed(0)
    module = RotaryEmbedding(channels=8, layout=layout)
    reference = RotaryEmbedding(channels=8, layout=layout)

    module.to(torch.bfloat16)
    assert module.inv_freq.dtype == torch.float32
    # Values survive the round-trip exactly (not just the dtype): the cast
    # itself already rounds, so the originals are restored, not upcast.
    torch.testing.assert_close(
        module.inv_freq,
        reference.inv_freq,
        atol=0.0,
        rtol=0.0,
    )

    x = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
    out = module(x)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(
        out.float(),
        reference(x.float()),
        atol=1e-2,
        rtol=1e-2,
    )
