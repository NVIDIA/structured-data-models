# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import torch.nn.functional as F

from sdm.testing import onlyCUDA


@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_rmsnorm_cast(
    dtype: torch.dtype,
    affine: bool,
    strided: bool,
) -> None:
    module = importlib.import_module("sdm._kernels.triton.rmsnorm_cast")
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

    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype),
    ):
        expected = F.rms_norm(x, (channels,), weight, eps).to(dtype)
        actual = module.rmsnorm_cast(x, weight, eps)

    assert torch.cuda.current_device() == current_device
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )
