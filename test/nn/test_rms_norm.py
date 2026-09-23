# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.nn import RMSNorm
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
@pytest.mark.parametrize("affine", [False, True])
def test_rms_norm(device: torch.device, affine: bool) -> None:
    reference = torch.nn.RMSNorm(64, elementwise_affine=affine, device=device)
    if reference.weight is not None:
        with torch.no_grad():
            reference.weight.uniform_(-2, 2)
    norm = RMSNorm(64, elementwise_affine=affine, device=device)
    norm.load_state_dict(reference.state_dict())
    reference.load_state_dict(norm.state_dict())
    x = torch.randn(2, 17, 64, device=device)

    with torch.inference_mode():
        actual = norm(x)
        expected = reference(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
@pytest.mark.parametrize("channels", [32, 64, 128, 256, 512])
@pytest.mark.parametrize(("affine", "eps"), [(False, 1e-6), (True, None)])
def test_rms_norm_autocast(
    channels: int,
    affine: bool,
    eps: float | None,
) -> None:
    reference = torch.nn.RMSNorm(
        channels, eps=eps, elementwise_affine=affine, device="cuda"
    )
    if reference.weight is not None:
        with torch.no_grad():
            reference.weight.uniform_(-2, 2)
    norm = RMSNorm(channels, eps=eps, elementwise_affine=affine, device="cuda")
    norm.load_state_dict(reference.state_dict())
    x = torch.randn(2, 257, channels, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 0
    x[0, 1] *= 1e-6

    with torch.inference_mode(), torch.autocast("cuda", dtype=x.dtype):
        actual = norm(x)
        expected = reference(x).to(x.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
def test_rms_norm_multidimensional(device: torch.device) -> None:
    reference = torch.nn.RMSNorm([2, 32], device=device)
    norm = RMSNorm([2, 32], device=device)
    norm.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, 32, device=device)

    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16),
    ):
        actual = norm(x)
        expected = reference(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
def test_rms_norm_rejects_wrong_shape(device: torch.device) -> None:
    norm = RMSNorm(64, elementwise_affine=False, device=device)
    x = torch.randn(2, 32, device=device)

    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16),
        pytest.raises(RuntimeError, match="normalized_shape"),
    ):
        norm(x)
