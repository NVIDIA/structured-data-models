# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pytest
import torch

from sdm.nn import RMSNorm, RotaryEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(("affine", "eps"), [(False, 1e-6), (True, None)])
@pytest.mark.parametrize(
    ("normalized_shape", "dtype", "autocast"),
    [
        (64, torch.float32, False),
        ([2, 32], torch.float32, True),
        (32, torch.bfloat16, True),
        (64, torch.bfloat16, True),
        (128, torch.bfloat16, True),
        (256, torch.bfloat16, True),
        (512, torch.bfloat16, True),
    ],
)
def test_rms_norm(
    device: torch.device,
    affine: bool,
    eps: float | None,
    normalized_shape: int | list[int],
    dtype: torch.dtype,
    autocast: bool,
) -> None:
    reference = torch.nn.RMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=affine,
        device=device,
    )
    if reference.weight is not None:
        with torch.no_grad():
            reference.weight.uniform_(-2, 2)
    norm = RMSNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        elementwise_affine=affine,
        device=device,
    )
    norm.load_state_dict(reference.state_dict())
    reference.load_state_dict(norm.state_dict())
    x = torch.randn(
        2, 17, *reference.normalized_shape, device=device, dtype=dtype
    )
    x[0, 0] = 0
    x[0, 1] *= 1e-6

    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast),
    ):
        actual = norm(x)
        expected = reference(x)
    if (
        autocast
        and device.type == "cuda"
        and isinstance(normalized_shape, int)
    ):
        expected = expected.to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
@pytest.mark.parametrize(
    ("layout", "partial_rotary_factor"),
    [("split_half", 1.0), ("split_half", 0.5), ("interleaved", 1.0)],
)
@pytest.mark.parametrize(
    ("dtype", "autocast"),
    [
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.float32, True),
        (torch.float32, False),
    ],
)
@pytest.mark.parametrize("channels", [24, 32, 64])
def test_rms_norm_rope(
    device: torch.device,
    layout: Literal["split_half", "interleaved"],
    partial_rotary_factor: float,
    dtype: torch.dtype,
    autocast: bool,
    channels: int,
) -> None:
    rope = RotaryEmbedding(
        channels=channels,
        layout=layout,
        partial_rotary_factor=partial_rotary_factor,
        device=device,
    )
    norm = RMSNorm(channels, eps=1e-6, elementwise_affine=False, device=device)
    x = torch.randn(2, 17, 3, channels, device=device, dtype=dtype)
    autocast_dtype = dtype if dtype != torch.float32 else torch.float16

    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast),
    ):
        actual = norm(x, rope=rope)
        expected = norm(rope(x))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
def test_rms_norm_rope_grad(device: torch.device) -> None:
    rope = RotaryEmbedding(channels=32, layout="split_half", device=device)
    norm = RMSNorm(32, eps=1e-6, device=device)
    x = torch.randn(2, 5, 3, 32, device=device, requires_grad=True)

    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = norm(x, rope=rope)
        expected = norm(rope(x))

    inputs = (x, rope.inv_freq, norm.weight)
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)


@withCUDA
def test_rms_norm_rope_empty(device: torch.device) -> None:
    rope = RotaryEmbedding(channels=32, layout="split_half", device=device)
    norm = RMSNorm(32, eps=1e-6, device=device)
    x = torch.empty(2, 0, 3, 32, device=device)

    with torch.inference_mode(), torch.autocast(device.type):
        actual = norm(x, rope=rope)
        expected = norm(rope(x))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
