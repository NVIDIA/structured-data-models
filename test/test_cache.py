# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from typing import cast

import pytest
import torch

from sdm.cache import Cache, KVCacheEntry, QuantizedKVCacheEntry
from sdm.testing import onlyCUDA, withCUDA


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
@pytest.mark.parametrize("map_to_device", [False, True])
def test_cache_serialization(
    device: torch.device,
    map_to_device: bool,
) -> None:
    nested = Cache(
        entry=KVCacheEntry(
            key=torch.arange(6).view(2, 3), value=torch.ones(2, 3)
        )
    )
    if device.type == "cuda":
        nested = nested.pin_memory()
    cache = Cache(foo="foo", tensor=torch.arange(3), nested=nested).freeze()

    stream = io.BytesIO()
    torch.save(cache, stream)
    stream.seek(0)
    restored = torch.load(
        stream,
        map_location=device if map_to_device else None,
        weights_only=False,
    )

    assert restored is not cache
    assert restored["foo"] == "foo"
    assert restored.is_replaying
    restored_nested = cast(Cache, restored["nested"])
    assert restored_nested.is_replaying
    assert restored_nested.is_pinned() == (
        device.type == "cuda" and not map_to_device
    )
    assert not cast(torch.Tensor, restored["tensor"]).is_pinned()

    load_device = device if map_to_device else torch.device("cpu")
    expected_entry = cast(KVCacheEntry, nested["entry"])
    restored_entry = cast(KVCacheEntry, restored_nested["entry"])
    for actual, expected in (
        (
            cast(torch.Tensor, restored["tensor"]),
            cast(torch.Tensor, cache["tensor"]),
        ),
        (restored_entry.key, expected_entry.key),
        (restored_entry.value, expected_entry.value),
    ):
        assert actual.device == load_device
        torch.testing.assert_close(
            actual, expected.to(load_device), rtol=0, atol=0
        )


@onlyCUDA
def test_cache_serialization_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache(tensor=torch.arange(3).pin_memory())
    stream = io.BytesIO()
    torch.save(cache, stream)
    stream.seek(0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    restored = torch.load(stream, map_location="cpu", weights_only=False)

    assert restored.is_recording
    assert restored.is_cpu
    assert not restored.is_pinned()
    torch.testing.assert_close(
        restored["tensor"], cache["tensor"], rtol=0, atol=0
    )


@withCUDA
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [
        (torch.int8, 0.5 / 127),
        (torch.float8_e4m3fn, 2**-4),
        (torch.float8_e5m2, 2**-3),
    ],
)
def test_quantized_kv_cache_entry(
    device: torch.device, dtype: torch.dtype, tolerance: float
) -> None:
    entry = KVCacheEntry(
        key=torch.randn(2, 5, 3, 8, device=device) * 100,
        value=torch.randn(2, 5, 3, 8, device=device),
    )
    entry.key[0, 0, 0] = 0

    quantized = QuantizedKVCacheEntry.from_entry(entry, dtype)
    assert quantized.key.dtype == quantized.value.dtype == dtype
    assert quantized.key_scale.size() == (2, 5, 3, 1)

    restored = quantized.dequantize()
    assert restored.dequantize() is restored
    for actual, expected in (
        (restored.key, entry.key),
        (restored.value, entry.value),
    ):
        assert actual.dtype == torch.float32
        absmax = expected.abs().amax(-1, keepdim=True)
        assert ((actual - expected).abs() <= tolerance * absmax).all()
    assert (restored.key[0, 0, 0] == 0).all()


@withCUDA
def test_cache_kv_cache_dtype(device: torch.device) -> None:
    entry = KVCacheEntry(
        key=torch.randn(4, 2, 8, device=device, dtype=torch.float16),
        value=torch.randn(4, 2, 8, device=device, dtype=torch.float16),
    )
    cache = Cache(entry=entry, kv_cache_dtype=torch.int8)
    cache["tensor"] = entry.key
    cache["nested"] = Cache(entry=entry)

    assert isinstance(cache["entry"], QuantizedKVCacheEntry)
    assert cache["tensor"] is entry.key
    # INT8 payloads and FP32 scales, plus the uncompressed tensor and entry.
    assert cache.size() == 2 * (64 + 4 * 8) + 64 * 2 + 2 * 64 * 2

    cache = cache.cpu()
    assert cache.kv_cache_dtype == torch.int8
    assert cast(QuantizedKVCacheEntry, cache["entry"]).key_scale.is_cpu
    cache["other"] = entry
    assert isinstance(cache["other"], QuantizedKVCacheEntry)

    stream = io.BytesIO()
    torch.save(cache.freeze(), stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    torch.testing.assert_close(
        cast(KVCacheEntry, restored["entry"]).dequantize().key,
        cast(KVCacheEntry, cache["entry"]).dequantize().key,
        rtol=0,
        atol=0,
    )
