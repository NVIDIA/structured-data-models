# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from typing import cast

import pytest
import torch

from sdm.cache import (
    Cache,
    Float8PerHead,
    KVCacheEntry,
    QuantizedTensor,
    Unquantized,
)
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


def test_kv_cache_entry_quantized() -> None:
    x = torch.ones(2, 3, 1, 4)  # [B, KV, H, C]
    entry = KVCacheEntry(key=Float8PerHead().quantize(x), value=x)
    cache = Cache(entry=entry).cpu()
    assert isinstance(cast(KVCacheEntry, cache["entry"]).key, QuantizedTensor)
    assert cache.size() == 24 * 1 + 2 * 4 + 24 * 4


def test_quantized_tensor_rejects_mismatched_scheme() -> None:
    qdata = torch.randn(2, 3, 1, 4).to(torch.float8_e4m3fn)
    scale = torch.ones(2, 1, 1, 1)
    fp8 = Float8PerHead()
    QuantizedTensor(qdata=qdata, scale=scale, scheme=fp8)
    with pytest.raises(ValueError, match="do not match"):
        QuantizedTensor(qdata=qdata.float(), scale=scale, scheme=fp8)
    with pytest.raises(ValueError, match="do not match"):
        QuantizedTensor(qdata=qdata, scale=scale[:1], scheme=fp8)
    with pytest.raises(ValueError, match="do not match"):
        QuantizedTensor(qdata=qdata, scale=scale, scheme=Unquantized())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cache_fp8_per_head(dtype: torch.dtype) -> None:
    x = torch.randn(2, 5, 3, 4, dtype=dtype)  # [B, KV, H, C]
    quantized = Float8PerHead().quantize(x)
    assert quantized.qdata.dtype == torch.float8_e4m3fn
    assert quantized.scale.size() == (2, 1, 3, 1)
    assert quantized.shape == x.shape

    out = quantized.dequantize(dtype)
    torch.testing.assert_close(out, x, rtol=2**-4, atol=1e-2)

    cache = Cache(kv_cache_dtype="fp8_per_head")
    assert isinstance(cache.kv_cache_dtype, Float8PerHead)
    with pytest.raises(ValueError, match="Unsupported key/value cache dtype"):
        Cache(kv_cache_dtype="unknown")
