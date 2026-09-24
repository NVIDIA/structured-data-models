# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sdm._kernels import rms_norm
from sdm.nn import RotaryEmbedding
from sdm.testing import onlyCUDA


def _expected(x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    out = F.rms_norm(x.float(), (x.size(-1),), weight, eps=1e-6)
    return out.half()


@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("channels", [32, 128, 512])
def test_rms_norm(dtype: torch.dtype, affine: bool, channels: int) -> None:
    x = torch.randn(3, 37, 5, channels, device="cuda").mul(4).to(dtype)
    weight = torch.randn(channels, device="cuda") if affine else None

    views = [
        x,
        x.transpose(1, 2),  # Strided leading dimensions.
        x[:, 1:, 2:],  # Offset slice.
        x.expand(2, *x.size())[..., :channels],  # Broadcast dimension.
    ]
    for view in views:
        out = rms_norm(view, weight, eps=1e-6, dtype=torch.float16)
        assert out.dtype == torch.float16
        assert out.size() == view.size()
        torch.testing.assert_close(out, _expected(view, weight))


@onlyCUDA
def test_rms_norm_heads() -> None:
    # Heads split off a fused projection: rows are not evenly strided.
    qkv = torch.randn(2, 11, 3 * 128, device="cuda", dtype=torch.float16)
    query = qkv[..., :128].unflatten(-1, (4, 32))

    out = rms_norm(query, None, eps=1e-6, dtype=torch.float16)

    torch.testing.assert_close(out, _expected(query, None))


@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rms_norm_rope(dtype: torch.dtype) -> None:
    rope = RotaryEmbedding(
        channels=32,
        layout="split_half",
        requires_grad=False,
        device="cuda",
    )
    qkv = torch.randn(3, 2, 70, 3 * 128, device="cuda").mul(4).to(dtype)
    query = qkv[..., :128].unflatten(-1, (4, 32))

    out = rms_norm(query, None, eps=1e-6, dtype=torch.float16, rope=rope)

    torch.testing.assert_close(out, _expected(rope(query), None))


def test_rms_norm_fallback() -> None:
    x = torch.randn(4, 6, 24)
    weight = torch.randn(24)

    out = rms_norm(x, weight, eps=1e-6, dtype=torch.float16)

    torch.testing.assert_close(out, _expected(x, weight))
