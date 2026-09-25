# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm.cache import Cache, Int8KVCacheEntry, KVCacheEntry
from sdm.testing import withCUDA


@withCUDA
def test_cache(device: torch.device) -> None:
    cache = Cache(foo="foo")
    assert len(cache) == 1
    cache["entry"] = KVCacheEntry(key=torch.randn(5), value=torch.randn(5))
    assert len(cache) == 2
    cache = cache.cpu()
    assert cast(KVCacheEntry, cache["entry"]).key.is_cpu
    assert cast(KVCacheEntry, cache["entry"]).value.is_cpu

    if device.type == "cuda":
        cache = cache.pin_memory()
        assert cast(KVCacheEntry, cache["entry"]).key.is_pinned()
        assert cast(KVCacheEntry, cache["entry"]).value.is_pinned()
        cache = cache.to(device, non_blocking=True)
        assert cast(KVCacheEntry, cache["entry"]).key.is_cuda
        assert cast(KVCacheEntry, cache["entry"]).value.is_cuda


def test_cache_size() -> None:
    cache = Cache(
        entry=KVCacheEntry(
            key=torch.ones(3, dtype=torch.float32),
            value=torch.ones(2, dtype=torch.int64),
        ),
        nested=[
            Cache(
                entry=KVCacheEntry(
                    key=torch.ones(5, dtype=torch.bool),
                    value=torch.ones(4, dtype=torch.float16),
                )
            )
        ],
    )

    assert cache.size() == 3 * 4 + 2 * 8 + 5 * 1 + 4 * 2


@withCUDA
@pytest.mark.parametrize(
    ("key_dtype", "value_dtype"),
    [(torch.float32, torch.float32), (torch.float16, torch.bfloat16)],
)
def test_int8_kv_cache_entry(
    device: torch.device, key_dtype: torch.dtype, value_dtype: torch.dtype
) -> None:
    entry = KVCacheEntry(
        key=torch.rand(2, 6, 3, 8, device=device, dtype=key_dtype) * 2 - 1,
        value=torch.rand(2, 6, 3, 8, device=device, dtype=value_dtype) * 2 - 1,
    )

    quantized = Int8KVCacheEntry.from_entry(entry)
    assert quantized.key.dtype == torch.int8
    assert quantized.value.dtype == torch.int8
    assert quantized.key_scale.size() == (2, 6, 3, 1)
    assert quantized.key_scale.dtype == torch.float32

    restored = quantized.dequantize()
    torch.testing.assert_close(restored.key, entry.key, atol=1e-2, rtol=0)
    torch.testing.assert_close(restored.value, entry.value, atol=1e-2, rtol=0)


@pytest.mark.parametrize("rows", [0, 2])
def test_int8_kv_cache_entry_constant(rows: int) -> None:
    entry = KVCacheEntry(
        key=torch.zeros(rows, 1, 4), value=torch.full((rows, 1, 4), 3.0)
    )
    restored = Int8KVCacheEntry.from_entry(entry).dequantize()
    torch.testing.assert_close(restored.key, entry.key)
    torch.testing.assert_close(restored.value, entry.value)


@withCUDA
def test_cache_quantizes_recorded_key_values(device: torch.device) -> None:
    entry = KVCacheEntry(
        key=torch.randn(4, 2, 8, device=device, dtype=torch.float16),
        value=torch.randn(4, 2, 8, device=device, dtype=torch.float16),
    )

    cache = Cache(kv_cache_dtype=torch.int8)
    cache["entry"] = entry
    assert isinstance(cache["entry"], Int8KVCacheEntry)

    default = Cache(entry=entry)
    assert default["entry"] is entry
    assert cache.size() < default.size()

    cache["other"] = entry.key
    assert cache["other"] is entry.key
    moved = cache.freeze().cpu()
    quantized = moved["entry"]
    assert isinstance(quantized, Int8KVCacheEntry)
    assert quantized.is_cpu
    torch.testing.assert_close(
        quantized.dequantize().key, entry.key.cpu(), atol=2e-2, rtol=0
    )

    derived = moved.new_empty()
    derived["entry"] = entry
    assert isinstance(derived["entry"], Int8KVCacheEntry)


def test_cache_rejects_unsupported_kv_cache_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported key/value cache dtype"):
        Cache(kv_cache_dtype=torch.float16)
