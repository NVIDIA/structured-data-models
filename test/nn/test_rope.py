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
@pytest.mark.parametrize("partial_rotary_factor", [0.5, 1.0])
def test_rope_position_index(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
    partial_rotary_factor: float,
) -> None:
    module = RotaryEmbedding(
        channels=8,
        layout=layout,
        partial_rotary_factor=partial_rotary_factor,
        device=device,
    )
    x = torch.randn(2, 3, 4, 2, 8, device=device)
    positions = torch.tensor(
        [[[-1, 0, 2.5, 2.5]], [[3, 0, 1, 3]]], device=device
    )
    actual = module(x, position_index=positions)
    recovered = module(actual, position_index=-positions)
    torch.testing.assert_close(recovered, x)
    torch.testing.assert_close(actual[..., 1, :, :], x[..., 1, :, :])
    if partial_rotary_factor < 1:
        torch.testing.assert_close(actual[..., 4:], x[..., 4:])
    contiguous = torch.arange(x.size(-3), device=device)
    torch.testing.assert_close(module(x), module(x, position_index=contiguous))
