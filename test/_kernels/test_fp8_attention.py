# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm._kernels.fp8_attention import fp8_attention
from sdm.cache import QuantizedKVCacheEntry
from sdm.testing import onlyCUDA, withCUDA


@withCUDA
@pytest.mark.parametrize("channels", [32, 48])
@pytest.mark.parametrize("grad", [False, True])
def test_fp8_dispatch(device: torch.device, channels: int, grad: bool) -> None:
    if device.type == "cuda" and (
        torch.version.cuda is None
        or torch.cuda.get_device_capability(device) < (8, 9)
    ):
        pytest.skip("FP8 requires CUDA compute capability 8.9 or newer")
    x = torch.ones(1, 8193, 2, channels, device=device, dtype=torch.float16)
    with torch.set_grad_enabled(grad):
        result = fp8_attention(x, x, x)
        if device.type == "cpu" or grad or channels == 48:
            assert result is None
            return
        assert result is not None
        output, cache = result
        torch.testing.assert_close(output, x)
        assert cache.key.dtype == torch.float8_e4m3fn
        cached = fp8_attention(x[:, :3], cache.key, cache.value, cache)
        assert cached is not None
        torch.testing.assert_close(cached[0], x[:, :3])
        empty = fp8_attention(x[:, :0], cache.key, cache.value, cache)
        assert empty is not None
        assert empty[0].shape == x[:, :0].shape
        assert empty[1] is cache


@onlyCUDA
@pytest.mark.parametrize(
    "capability", [(8, 6), (10, 0), (10, 3), (11, 0), (12, 1)]
)
def test_fp8_device_capability(
    capability: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.version.cuda or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 requires CUDA compute capability 8.9 or newer")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _: capability
    )
    x = torch.ones(1, 8193, 1, 32, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        result = fp8_attention(x, x, x)
    if capability < (8, 9):
        assert result is None
    else:
        assert result is not None
        torch.testing.assert_close(result[0], x)


@withCUDA
def test_fp8_cache_rejects_unsupported_execution(device: torch.device) -> None:
    x = torch.ones(1, 17, 2, 32, device=device, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, 1, 2, 1, device=device)
    cache = QuantizedKVCacheEntry(
        key=x,
        value=x,
        key_scale=scale,
        value_scale=scale,
        query_scale=scale,
        dtype=torch.float16,
    )
    query = torch.ones(1, 3, 2, 32, device=device, dtype=torch.float16)
    with (
        torch.enable_grad(),
        pytest.raises(ValueError, match="CUDA inference"),
    ):
        fp8_attention(query, cache.key, cache.value, cache)
