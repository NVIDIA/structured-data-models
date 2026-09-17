# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.kumo.timeseries.forecasting.attention import (
    CrossChannelAttention,
)
from sdm.models.kumo.timeseries.forecasting.head import ForecastingHead
from sdm.models.kumo.timeseries.forecasting.normalization import RevIN
from sdm.models.kumo.timeseries.forecasting.patch import (
    PatchEmbedding,
    Patching,
    patch_mask,
)


def test_patching() -> None:
    x = torch.arange(8).reshape(1, 1, 8)

    out = Patching(patch_len=4, stride=4)(x)

    expected = torch.tensor([[[[0, 1, 2, 3], [4, 5, 6, 7]]]])
    assert out.equal(expected)


def test_patch_mask_requires_every_time_step() -> None:
    mask = torch.tensor([[True, True, True, False, True, True]])

    out = patch_mask(mask, patch_len=3, stride=3)

    assert out.equal(torch.tensor([[True, False]]))


def test_patch_embedding_replaces_missing_patches() -> None:
    embedding = PatchEmbedding(
        patch_len=2,
        stride=2,
        channels=3,
        dropout=0.0,
        add_positional_embedding=False,
        bias=False,
        orthogonal_gain=None,
    )
    embedding.value_embedding.weight.data.fill_(1)
    embedding.mask_embedding.data.fill_(7)
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    mask = torch.tensor([[True, True, True, False]])

    out = embedding(x, mask)

    expected = torch.tensor([[[[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]]])
    assert out.equal(expected)


def test_revin_round_trip() -> None:
    normalizer = RevIN(num_features=1)
    x = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 100.0],
                [4.0, 5.0, 6.0, 100.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, True, False]])

    normalized, state = normalizer(x, mask)
    restored = normalizer.inverse(normalized, state)

    assert restored.allclose(x)
    assert normalized[..., :3].mean(dim=-1).abs().max() < 1e-6


def test_cross_channel_attention_is_permutation_equivariant() -> None:
    attention = CrossChannelAttention(
        channels=8,
        num_heads=2,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 3, 4, 8)
    permutation = torch.tensor([2, 0, 1])

    out = attention(x)
    permuted_out = attention(x[:, permutation])

    assert permuted_out.allclose(out[:, permutation], atol=1e-6)


def test_forecasting_head_shape() -> None:
    head = ForecastingHead(
        input_channels=4 * 8,
        prediction_length=6,
        dropout=0.0,
    )

    out = head(torch.randn(2, 3, 4, 8))

    assert out.size() == (2, 3, 6)


def test_layers_construct_on_meta_device() -> None:
    patch_embedding = PatchEmbedding(
        patch_len=8,
        stride=8,
        channels=16,
        device="meta",
    )
    normalizer = RevIN(num_features=1, affine=True, device="meta")
    attention = CrossChannelAttention(
        channels=16,
        num_heads=2,
        device="meta",
    )
    head = ForecastingHead(
        input_channels=32,
        prediction_length=6,
        device="meta",
    )

    assert patch_embedding.mask_embedding.device.type == "meta"
    assert normalizer.affine_weight.device.type == "meta"
    assert attention.attn.in_proj_weight.device.type == "meta"
    assert head.linear.weight.device.type == "meta"
