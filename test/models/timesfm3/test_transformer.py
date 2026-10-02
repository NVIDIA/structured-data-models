# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.models.timesfm3.transformer import (
    RotaryPositionalEmbedding,
    make_attn_mask,
)
from sdm.testing import withCUDA


@withCUDA
def test_make_attn_mask(device: torch.device) -> None:
    patch_mask = torch.tensor(
        [[True, False, False], [False, True, False]], device=device
    )
    mask = make_attn_mask(patch_mask)
    causal = torch.ones(3, 3, dtype=torch.bool, device=device).tril()
    expected = causal[None, None] & ~patch_mask[:, None, None, :]
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_attn_mask_noncausal(device: torch.device) -> None:
    patch_mask = torch.tensor([[True, False, False]], device=device)
    mask = make_attn_mask(patch_mask, causal=False)
    torch.testing.assert_close(mask, ~patch_mask[:, None, None, :])


@withCUDA
@pytest.mark.parametrize("rank", [3, 4])
def test_rotary_positional_embedding(
    device: torch.device,
    rank: int,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    shape = (2, 3, 4) if rank == 3 else (2, 3, 2, 4)
    inputs = torch.arange(
        math.prod(shape),
        device=device,
        dtype=torch.float32,
    ).reshape(shape)
    output = rope(inputs)

    torch.testing.assert_close(
        output.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
    )
    torch.testing.assert_close(output[0, 0], inputs[0, 0])
    assert output.shape == inputs.shape
    assert output.device == device


def test_rotary_positional_embedding_excludes_buffer_from_checkpoint() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    assert rope.state_dict() == {}


def test_rotary_positional_embedding_rejects_wrong_dimension() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    with pytest.raises(ValueError, match="must match the hidden dimension"):
        rope(torch.zeros(1, 2, 6))


def test_rotary_positional_embedding_rejects_wrong_rank() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4)

    with pytest.raises(ValueError, match="rank 3 or 4"):
        rope(torch.zeros(2, 4))


def test_rotary_positional_embedding_meta_device() -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device="meta")

    assert rope.timescale.device.type == "meta"


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_positional_embedding_rotation(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    inputs = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]],
        device=device,
        dtype=dtype,
    )

    output = rope(inputs)

    expected = torch.tensor(
        [math.cos(1.0), math.cos(0.01), math.sin(1.0), math.sin(0.01)],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(output[0, 1], expected)
