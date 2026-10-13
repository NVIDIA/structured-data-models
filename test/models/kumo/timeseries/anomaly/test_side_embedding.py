# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("is_unconditional", [False, True])
@pytest.mark.parametrize("time_channels", [8, 128])
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 7)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_side_embedding(
    device: torch.device,
    is_unconditional: bool,
    time_channels: int,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
    """Match Kumo-TS time/feature conditioning, mask order, and gradients."""
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA bfloat16 not supported")
    batch, features, length = shape
    module = SideEmbedding(
        num_features=features,
        time_channels=time_channels,
        feature_channels=4,
        is_unconditional=is_unconditional,
        device=device,
        dtype=dtype,
    )
    time = torch.arange(batch * length, device=device).reshape(batch, length)
    time = time.float().square() * 0.25 - 2.5
    mask = torch.arange(batch * features * length, device=device)
    mask = mask.reshape(shape).remainder(2).bool()
    actual = module(time, mask)

    # Upstream interleaves sin/cos, then normalizes time and feature channels.
    frequencies = 1 / torch.pow(
        10000.0,
        torch.arange(0, time_channels, 2, device=device) / time_channels,
    )
    phase = time.unsqueeze(-1) * frequencies
    time_embed = torch.empty(batch, length, time_channels, device=device)
    time_embed[..., 0::2] = phase.sin()
    time_embed[..., 1::2] = phase.cos()
    time_embed = F.layer_norm(
        input=time_embed.to(dtype),
        normalized_shape=(time_channels,),
        weight=module.time_embed_norm.weight,
        bias=module.time_embed_norm.bias,
        eps=module.time_embed_norm.eps,
    )
    feature_embed = F.embedding(
        torch.arange(features, device=device), module.embed_layer.weight
    )
    feature_embed = F.layer_norm(
        input=feature_embed,
        normalized_shape=(4,),
        weight=module.embed_norm.weight,
        bias=module.embed_norm.bias,
        eps=module.embed_norm.eps,
    )
    expected = torch.cat(
        [
            time_embed[:, :, None].expand(-1, -1, features, -1),
            feature_embed[None, None].expand(batch, length, -1, -1),
        ],
        dim=-1,
    ).permute(0, 3, 2, 1)
    if is_unconditional:
        expected = torch.cat([expected, mask[:, None].to(dtype)], dim=1)
    assert actual.shape == (
        batch,
        time_channels + 4 + int(is_unconditional),
        features,
        length,
    )
    assert actual.dtype == dtype
    assert actual.device == device
    torch.testing.assert_close(actual, expected)
    actual.float().square().sum().backward()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()

    changed = module(time, ~mask)
    if is_unconditional:
        torch.testing.assert_close(changed[:, :-1], actual[:, :-1])
        torch.testing.assert_close(changed[:, -1], (~mask).to(dtype))
    else:
        torch.testing.assert_close(changed, actual)


@withCUDA
def test_side_embedding_conversion_and_state_dict(
    device: torch.device,
) -> None:
    """Preserve conditioning after dtype conversion and state loading."""
    module = SideEmbedding(
        num_features=3,
        time_channels=8,
        feature_channels=4,
        is_unconditional=True,
    ).to(device=device, dtype=torch.float64)
    time = torch.tensor([[0, 1, 100000]], device=device)
    mask = torch.ones(1, 3, 3, device=device, dtype=torch.float32)
    expected = module(time, mask)
    restored = SideEmbedding(
        num_features=3,
        time_channels=8,
        feature_channels=4,
        is_unconditional=True,
        device=device,
        dtype=torch.float64,
    )
    restored.load_state_dict(module.state_dict(), strict=True)
    actual = restored(time, mask)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, -1], mask.double())
