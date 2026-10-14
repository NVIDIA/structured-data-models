# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sdm.models.kumo.timeseries.anomaly.embedding import SideEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    ("is_unconditional", "time_channels", "shape", "dtype"),
    [
        (False, 8, (2, 3, 7), torch.float32),
        (True, 128, (2, 3, 7), torch.float32),
        (False, 128, (1, 1, 1), torch.float32),
        (True, 8, (1, 1, 1), torch.float32),
        (False, 8, (2, 3, 7), torch.float16),
        (True, 8, (2, 3, 7), torch.bfloat16),
        (True, 8, (2, 3, 7), torch.float64),
    ],
)
def test_side_embedding(
    device: torch.device,
    is_unconditional: bool,
    time_channels: int,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> None:
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
        dtype=None if dtype == torch.float64 else dtype,
    )
    if dtype == torch.float64:
        module = module.to(dtype=dtype)
    time = torch.arange(batch * length, device=device).reshape(batch, length)
    time = time.float().square() * 0.25 - 2.5
    time[:, -1] = 100000
    mask = torch.arange(batch * features * length, device=device)
    mask = mask.reshape(shape).remainder(2).bool()
    actual = module(time, mask)

    # Upstream interleaves sin/cos, then normalizes time and feature channels.
    frequencies = 1 / torch.pow(
        self=10000.0,
        exponent=torch.arange(0, time_channels, 2, device=device)
        / time_channels,
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
        input=torch.arange(features, device=device),
        weight=module.embed_layer.weight,
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
    if dtype == torch.float64:
        actual.square().sum().backward()
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert parameter.grad.isfinite().all()

    changed = module(time, ~mask)
    if is_unconditional:
        torch.testing.assert_close(changed[:, :-1], actual[:, :-1])
        torch.testing.assert_close(changed[:, -1], (~mask).to(dtype))
    else:
        torch.testing.assert_close(changed, actual)
