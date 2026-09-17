# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from sdm.models.timesfm3.transformer import (
    RotaryPositionalEmbedding,
    make_attn_mask,
    make_segment_mask,
)
from sdm.testing import withCUDA


@withCUDA
def test_make_attn_mask(device: torch.device) -> None:
    num_masked = torch.tensor([1, 2], device=device)
    query_offset = torch.tensor([0, 2], device=device)

    mask = make_attn_mask(
        query_length=2,
        num_all_masked_kv=num_masked,
        query_index_offset=query_offset,
        kv_length=4,
    )

    expected = torch.tensor(
        [
            [[[False, False, False, False], [False, True, False, False]]],
            [[[False, False, True, False], [False, False, True, True]]],
        ],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_attn_mask_noncausal(device: torch.device) -> None:
    num_masked = torch.tensor([2], device=device)

    mask = make_attn_mask(
        query_length=3,
        num_all_masked_kv=num_masked,
        kv_length=4,
        causal=False,
    )

    expected = torch.tensor(
        [[[[False, False, True, True]]]],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_make_segment_mask(device: torch.device) -> None:
    segment_ids = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 1, 2]],
        device=device,
    )

    mask = make_segment_mask(segment_ids)

    expected = segment_ids[:, :, None] == segment_ids[:, None, :]
    torch.testing.assert_close(mask, expected.unsqueeze(1))


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
    positions = torch.tensor([[0, 1, 2], [2, 3, 4]], device=device)

    output = rope(inputs, position=positions)

    torch.testing.assert_close(
        output.square().sum(dim=-1),
        inputs.square().sum(dim=-1),
    )
    torch.testing.assert_close(output[0, 0], inputs[0, 0])
    assert output.shape == inputs.shape
    assert output.device == device


@withCUDA
def test_rotary_positional_embedding_uses_sequential_positions(
    device: torch.device,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=4, device=device)
    inputs = torch.arange(24, device=device, dtype=torch.float32).reshape(
        2, 3, 4
    )
    positions = torch.arange(3, device=device).unsqueeze(0)

    torch.testing.assert_close(rope(inputs), rope(inputs, position=positions))


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
def test_make_attn_mask_uses_query_length_by_default(
    device: torch.device,
) -> None:
    num_masked = torch.tensor([1], device=device)

    mask = make_attn_mask(query_length=3, num_all_masked_kv=num_masked)

    expected = torch.tensor(
        [
            [
                [
                    [False, False, False],
                    [False, True, False],
                    [False, True, True],
                ]
            ]
        ],
        device=device,
    )
    torch.testing.assert_close(mask, expected)


@withCUDA
def test_rotary_positional_embedding_rotation_direction(
    device: torch.device,
) -> None:
    rope = RotaryPositionalEmbedding(embedding_dims=2, device=device)
    inputs = torch.tensor([[[1.0, 0.0]]], device=device)
    positions = torch.tensor([[1]], device=device)

    output = rope(inputs, position=positions)

    expected = torch.tensor(
        [[[math.cos(1.0), math.sin(1.0)]]],
        device=device,
    )
    torch.testing.assert_close(output, expected)
