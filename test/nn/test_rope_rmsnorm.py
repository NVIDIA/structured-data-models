# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Literal

import pytest
import torch
from torch.nn import RMSNorm, Sequential

from sdm.nn import RotaryEmbedding
from sdm.nn._rope_rmsnorm import _RoPERMSNorm
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize(
    ("channels", "layout", "partial"),
    [
        (32, "split_half", 1.0),
        (64, "interleaved", 0.5),
    ],
)
@pytest.mark.parametrize("autocast", [False, True])
def test_rope_rmsnorm(
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
    layout: Literal["split_half", "interleaved"],
    partial: float,
    autocast: bool,
) -> None:
    native = Sequential(
        RotaryEmbedding(
            channels,
            layout=layout,
            partial_rotary_factor=partial,
            device=device,
        ),
        RMSNorm(channels, eps=1e-6, elementwise_affine=False, device=device),
    ).eval()
    fused = _RoPERMSNorm(*copy.deepcopy(native)).eval()
    packed = torch.randn(2, 3, 17, 4, 3 * channels, device=device, dtype=dtype)
    x = packed[..., :channels]
    with (
        torch.inference_mode(),
        torch.autocast(
            device.type,
            enabled=autocast and device.type == "cuda",
            dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
        ),
    ):
        torch.testing.assert_close(fused(x), native(x), rtol=0, atol=0)
    assert fused.state_dict().keys() == native.state_dict().keys()


@withCUDA
def test_rope_rmsnorm_gradients(device: torch.device) -> None:
    native = Sequential(
        RotaryEmbedding(32, layout="split_half", device=device),
        RMSNorm(32, eps=1e-6, elementwise_affine=False, device=device),
    )
    fused = _RoPERMSNorm(*copy.deepcopy(native)).eval()
    x = torch.randn(2, 17, 4, 32, device=device, requires_grad=True)
    expected = native(x)
    actual = fused(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    assert x.grad is not None
    grad = x.grad.clone()
    x.grad = None
    actual.sum().backward()
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
    torch.testing.assert_close(
        fused[0].inv_freq.grad, native[0].inv_freq.grad, rtol=0, atol=0
    )
