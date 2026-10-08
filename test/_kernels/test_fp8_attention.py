# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm._kernels.fp8_attention import fp8_attention
from sdm.cache import QuantizedKVCacheEntry
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("channels", [32, 48])
@pytest.mark.parametrize("grad", [False, True])
def test_fp8_dispatch(device: torch.device, channels: int, grad: bool) -> None:
    if device.type == "cuda" and torch.cuda.get_device_capability(
        device
    ) not in {(8, 9), (9, 0), (12, 0)}:
        pytest.skip("FP8 dispatch requires Ada, Hopper, or RTX Blackwell")
    x = torch.ones(1, 17, 2, channels, device=device, dtype=torch.float16)
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
