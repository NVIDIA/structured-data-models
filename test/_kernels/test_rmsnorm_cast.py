# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch.nn.functional as F

from sdm._kernels import rmsnorm_cast
from sdm.nn import RotaryEmbedding
from sdm.testing import onlyCUDA, withCUDA


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("channels", [32, 64])
def test_rmsnorm_cast(
    dtype: torch.dtype,
    affine: bool,
    strided: bool,
    channels: int,
) -> None:
    current_device = torch.cuda.current_device()
    device = torch.device(
        "cuda", (current_device + 1) % torch.cuda.device_count()
    )
    x = torch.randn(
        2,
        4097,
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


@onlyCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("channels", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("strided", [False, True])
def test_rmsnorm_cast_rope(
    dtype: torch.dtype,
    channels: int,
    strided: bool,
) -> None:
    rope = RotaryEmbedding(
        channels=channels,
        layout="split_half",
        requires_grad=False,
        device="cuda",
    )
    x = torch.randn(
        2,
        3,
        17,
        4,
        channels * (2 if strided else 1),
        device="cuda",
        dtype=dtype,
    )
    if strided:
        x = x[..., ::2]
    weight = torch.randn(channels, device=x.device)
    seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
    freq = seq[:, None] * rope.inv_freq[None, :]
    tables = (freq.cos().to(dtype), freq.sin().to(dtype))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        actual = rmsnorm_cast(x, weight, 1e-6, rope=tables)
        expected = F.rms_norm(rope(x), (channels,), weight, 1e-6).to(
            torch.float16
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@withCUDA
def test_rmsnorm_cast_rope_grad_fallback(device: torch.device) -> None:
    x = torch.randn(2, 5, 3, 64, device=device)
    freq = torch.randn(5, 32, device=device, requires_grad=True)
    tables = (freq.cos(), freq.sin())
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = rmsnorm_cast(x, None, 1e-6, rope=tables)
        cos, sin = (table.unsqueeze(-2) for table in tables)
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)
        expected = F.rms_norm(rotated, (64,), None, 1e-6).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual.sum(), freq, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), freq)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
