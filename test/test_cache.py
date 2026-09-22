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
def test_int8_kv_cache_entry(device: torch.device) -> None:
    entry = KVCacheEntry(
        key=torch.randn(2, 6, 3, 8, device=device),
        value=torch.randn(2, 6, 3, 8, device=device),
    )

    quantized = Int8KVCacheEntry.from_entry(entry)
    assert quantized.key.dtype == torch.int8
    assert quantized.value.dtype == torch.int8
    assert quantized.key.size() == entry.key.size()
    assert quantized.key_scale.size() == (2, 6, 3, 1)
    assert quantized.key_scale.dtype == torch.float32

    restored = quantized.dequantize()
    assert restored.key.dtype == entry.key.dtype
    assert restored.key.size() == entry.key.size()
    torch.testing.assert_close(restored.key, entry.key, atol=2e-2, rtol=0)
    torch.testing.assert_close(restored.value, entry.value, atol=2e-2, rtol=0)


def test_int8_kv_cache_entry_preserves_dtype() -> None:
    entry = KVCacheEntry(
        key=torch.randn(4, 2, 8, dtype=torch.float16),
        value=torch.randn(4, 2, 8, dtype=torch.bfloat16),
    )

    restored = Int8KVCacheEntry.from_entry(entry).dequantize()
    assert restored.key.dtype == torch.float16
    assert restored.value.dtype == torch.bfloat16


def test_int8_kv_cache_entry_constant_and_empty() -> None:
    constant = Int8KVCacheEntry.from_entry(
        KVCacheEntry(
            key=torch.zeros(2, 1, 4), value=torch.full((2, 1, 4), 3.0)
        )
    ).dequantize()
    assert (constant.key == 0).all()
    torch.testing.assert_close(
        constant.value, torch.full((2, 1, 4), 3.0), atol=2e-2, rtol=0
    )

    empty = Int8KVCacheEntry.from_entry(
        KVCacheEntry(key=torch.randn(0, 2, 4), value=torch.randn(0, 2, 4))
    ).dequantize()
    assert empty.key.size() == (0, 2, 4)


def test_cache_quantizes_recorded_key_values() -> None:
    entry = KVCacheEntry(
        key=torch.randn(4, 2, 8),
        value=torch.randn(4, 2, 8),
    )

    cache = Cache(kv_cache_dtype=torch.int8)
    cache["entry"] = entry
    assert isinstance(cache["entry"], Int8KVCacheEntry)

    # Non-key/value data and the default configuration are left untouched.
    cache["other"] = torch.randn(4)
    assert isinstance(cache["other"], torch.Tensor)
    default = Cache()
    default["entry"] = entry
    assert isinstance(default["entry"], KVCacheEntry)

    assert cache.size() < default.size()


def test_cache_rejects_unsupported_kv_cache_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported key/value cache dtype"):
        Cache(kv_cache_dtype=torch.float16)


@withCUDA
def test_cache_preserves_quantization_across_moves(
    device: torch.device,
) -> None:
    cache = Cache(kv_cache_dtype=torch.int8)
    cache["entry"] = KVCacheEntry(
        key=torch.randn(4, 2, 8, device=device),
        value=torch.randn(4, 2, 8, device=device),
    )

    moved = cache.cpu()
    entry = cast(Int8KVCacheEntry, moved["entry"])
    assert isinstance(entry, Int8KVCacheEntry)
    assert entry.key.is_cpu
    assert entry.key_scale.is_cpu
    assert entry.key_dtype == torch.float32

    # A derived cache keeps recording under the configured dtype.
    derived = moved.new_empty()
    derived["entry"] = KVCacheEntry(
        key=torch.randn(4, 2, 8), value=torch.randn(4, 2, 8)
    )
    assert isinstance(derived["entry"], Int8KVCacheEntry)


@pytest.mark.parametrize(
    ("channels_per_head", "expected"),
    [(2, "larger"), (4, "equal"), (8, "smaller")],
)
def test_int8_storage_break_even(
    channels_per_head: int, expected: str
) -> None:
    # One FP32 scale per token and head buys one byte per channel, so INT8
    # only shrinks a 16-bit cache once a head has more than four channels.
    entry = KVCacheEntry(
        key=torch.randn(6, 3, channels_per_head, dtype=torch.float16),
        value=torch.randn(6, 3, channels_per_head, dtype=torch.float16),
    )

    native = Cache()
    native["entry"] = entry
    quantized = Cache(kv_cache_dtype=torch.int8)
    quantized["entry"] = entry

    if expected == "larger":
        assert quantized.size() > native.size()
    elif expected == "equal":
        assert quantized.size() == native.size()
    else:
        assert quantized.size() < native.size()
