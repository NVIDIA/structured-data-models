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


@withCUDA
@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
def test_rope_partial_channels(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
) -> None:
    partial = RotaryEmbedding(
        channels=16,
        layout=layout,
        theta=100,
        rotary_channels=4,
        device=device,
    )
    reference = RotaryEmbedding(
        channels=4,
        layout=layout,
        theta=100,
        device=device,
    )
    x = torch.arange(
        2 * 5 * 3 * 16,
        device=device,
        dtype=torch.float32,
    ).reshape(2, 5, 3, 16)

    output = partial(x)
    expected_prefix = reference(x[..., :4])

    torch.testing.assert_close(output[..., :4], expected_prefix)
    assert torch.equal(output[..., 4:], x[..., 4:])
    with pytest.raises(ValueError, match="Expected 16 channels"):
        partial(x[..., :8])


@pytest.mark.parametrize("rotary_channels", [0, 3, 18])
def test_rope_invalid_partial_channels(rotary_channels: int) -> None:
    with pytest.raises(
        ValueError,
        match="'rotary_channels' must be an even number",
    ):
        RotaryEmbedding(
            channels=16,
            layout="split_half",
            rotary_channels=rotary_channels,
        )
