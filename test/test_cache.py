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
@pytest.mark.parametrize("quantized", [False, True])
def test_cache_head_selection_and_transfer(
    device: torch.device, quantized: bool
) -> None:
    key = torch.arange(192).reshape(2, 3, 4, 8).float()
    entry = (
        QuantizedKVCacheEntry(
            key=key.to(torch.float8_e4m3fn),
            value=(key / 2).to(torch.float8_e4m3fn),
            key_scale=torch.arange(8).reshape(2, 1, 4, 1).float() + 1,
            value_scale=torch.ones(2, 1, 4, 1),
            query_scale=torch.ones(2, 1, 8, 1),
            dtype=torch.float32,
        )
        if quantized
        else KVCacheEntry(key, key / 2)
    )
    selected = entry.select_heads(2)
    restored = selected.to(device).cpu()
    assert isinstance(restored, KVCacheEntry)
    assert entry.key.size(-2) == 4
    for name in ("key", "value"):
        actual = getattr(restored, name)
        expected = getattr(entry, name)[..., :2, :]
        torch.testing.assert_close(actual.float(), expected.float())
        assert actual.dtype == expected.dtype
        assert (
            actual.untyped_storage().nbytes()
            == actual.numel() * actual.element_size()
        )
    expected_bytes = restored.key.numel() * restored.key.element_size() * 2
    if isinstance(entry, QuantizedKVCacheEntry):
        assert isinstance(restored, QuantizedKVCacheEntry)
        assert restored.dtype == entry.dtype
        torch.testing.assert_close(
            restored.key_scale, entry.key_scale[..., :2, :]
        )
        torch.testing.assert_close(
            restored.value_scale, entry.value_scale[..., :2, :]
        )
        torch.testing.assert_close(restored.query_scale, entry.query_scale)
        expected_bytes += (
            restored.key_scale.numel()
            + restored.value_scale.numel()
            + restored.query_scale.numel()
        ) * 4
    assert Cache(entry=restored).size() == expected_bytes
