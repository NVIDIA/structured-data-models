# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch.nn.functional as F

from sdm._kernels import rmsnorm_cast
from sdm.testing import onlyCUDA, withCUDA


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_rmsnorm_cast(
    dtype: torch.dtype,
    affine: bool,
    strided: bool,
) -> None:
    current_device = torch.cuda.current_device()
    device = torch.device(
        "cuda", (current_device + 1) % torch.cuda.device_count()
    )
    channels = 64
    x = torch.randn(
        2,
        17,
        channels * (2 if strided else 1),
        device=device,
        dtype=dtype,
    )
    if strided:
        x = x[..., ::2]
    x[0, 0] = 0
    x[0, 1] *= 1e-4
    x[0, 2, 0] = float("inf")
    x[0, 3, 0] = float("nan")
    weight = None
    if affine:
        weight = torch.empty(
            channels * (2 if strided else 1), device=device
        ).uniform_(-2, 2)
        if strided:
            weight = weight[::2]
    eps = 1e-6

    output_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=output_dtype),
    ):
        expected = F.rms_norm(x, (channels,), weight, eps).to(output_dtype)
        actual = rmsnorm_cast(x, weight, eps)

    assert torch.cuda.current_device() == current_device
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )


@withCUDA
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((2, 63), torch.bfloat16),
        ((2, 64), torch.float64),
        ((0, 64), torch.bfloat16),
    ],
)
def test_rmsnorm_cast_fallback(
    device: torch.device,
    shape: tuple[int, int],
    dtype: torch.dtype,
) -> None:
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(shape[-1], device=device, dtype=dtype)
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=False),
    ):
        actual = rmsnorm_cast(x, weight, 1e-6)
        expected = F.rms_norm(x, (shape[-1],), weight, 1e-6).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
def test_rmsnorm_cast_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdm._kernels.rmsnorm_cast")
    monkeypatch.setattr(module, "_triton_rmsnorm_cast", None)
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=x.dtype):
        actual = rmsnorm_cast(x, weight, 1e-6)
        expected = F.rms_norm(x, (64,), weight, 1e-6).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@onlyCUDA
def test_rmsnorm_cast_grad_fallback() -> None:
    x = torch.randn(
        2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(64, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=x.dtype):
        actual = rmsnorm_cast(x, weight, 1e-6)
        expected = F.rms_norm(x, (64,), weight, 1e-6).to(x.dtype)
    actual_grads = torch.autograd.grad(actual.sum(), (x, weight))
    expected_grads = torch.autograd.grad(expected.sum(), (x, weight))
    torch.testing.assert_close(actual_grads, expected_grads)
