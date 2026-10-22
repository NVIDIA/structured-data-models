# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
def test_partial_rope(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
) -> None:
    partial = RotaryEmbedding(
        channels=16,
        layout=layout,
        theta=100,
        partial_rotary_factor=0.25,
        device=device,
    )
    reference = RotaryEmbedding(
        channels=4,
        layout=layout,
        theta=100,
        device=device,
    )
    x = torch.randn(2, 5, 3, 16, device=device)

    output = partial(x)
    assert torch.equal(output[..., 4:], x[..., 4:])
    expected_prefix = reference(x[..., :4])
    torch.testing.assert_close(output[..., :4], expected_prefix)


@withCUDA
@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
@pytest.mark.parametrize("batched_positions", [False, True])
def test_rope_explicit_positions(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
    partial_rotary_factor: float,
    batched_positions: bool,
) -> None:
    module = RotaryEmbedding(
        channels=16,
        layout=layout,
        partial_rotary_factor=partial_rotary_factor,
        device=device,
    )
    positions = torch.tensor([0, 7, 2, 7, 3], device=device)
    if batched_positions:
        positions = torch.stack([positions, positions.flip(0)]).unsqueeze(1)
    # Repeat each vector across time so rotations can be selected from a
    # default sequential call. Positions broadcast over the second batch axis.
    x = torch.randn(2, 3, 1, 2, 16, device=device)
    sequential = module(x.expand(2, 3, 8, 2, 16))
    index = positions[..., None, None].expand(2, 3, 5, 2, 16)
    expected = sequential.gather(-3, index)
    out = module(x.expand(2, 3, 5, 2, 16), pos=positions)
    torch.testing.assert_close(out, expected)
    rotated = int(16 * partial_rotary_factor)
    torch.testing.assert_close(
        out[..., rotated:], x[..., rotated:].expand_as(out[..., rotated:])
    )


@withCUDA
@pytest.mark.parametrize("layout", ["split_half", "interleaved"])
def test_rope_negative_positions_invert_rotation(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
) -> None:
    module = RotaryEmbedding(
        channels=16,
        layout=layout,
        partial_rotary_factor=0.5,
        device=device,
    )
    x = torch.randn(2, 5, 3, 16, device=device, requires_grad=True)
    positions = torch.tensor(
        [[0, 2, -3, 2, 7], [-1, 4, 0, 0, 3]], device=device
    )
    rotated = module(x, pos=positions)
    recovered = module(rotated, pos=-positions)
    torch.testing.assert_close(recovered, x)
    rotated.sum().backward()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    assert module.inv_freq.grad is not None
    assert module.inv_freq.grad.isfinite().all()
