# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.nn import RMSNorm
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
