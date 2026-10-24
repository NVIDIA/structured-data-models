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
def test_quantized_cache_head_selection_and_transfer(
    device: torch.device,
) -> None:
    key = torch.arange(192).reshape(2, 3, 4, 8).to(torch.float8_e4m3fn)
    scale = torch.arange(8).reshape(2, 1, 4, 1).float() + 1
    entry = QuantizedKVCacheEntry(
        key=key,
        value=key,
        key_scale=scale,
        value_scale=scale,
        query_scale=torch.ones(2, 1, 8, 1),
        dtype=torch.float32,
    )
    selected = entry.select_heads(2).to(device)
    assert isinstance(selected, QuantizedKVCacheEntry)
    assert selected.dtype == entry.dtype
    assert entry.key.size(-2) == 4
    for name in ("key", "value", "key_scale", "value_scale", "query_scale"):
        expected = getattr(entry, name)
        if name != "query_scale":
            expected = expected[..., :2, :]
        actual = getattr(selected, name)
        assert actual.device == device
        assert actual.dtype == expected.dtype
        assert actual.is_contiguous()
        torch.testing.assert_close(actual.cpu().float(), expected.float())
    # 192 K/V bytes + 32 K/V scale bytes + 64 query scale bytes.
    assert Cache(entry=selected).size() == 288


def test_cache_head_selection_releases_unused_heads() -> None:
    entry = KVCacheEntry(torch.ones(1, 1, 4, 8), torch.zeros(1, 1, 4, 8))
    selected = entry.select_heads(2)
    assert selected.key.shape == selected.value.shape == (1, 1, 2, 8)
    assert selected.key.data_ptr() != entry.key.data_ptr()
    assert selected.value.data_ptr() != entry.value.data_ptr()
    assert selected.key.is_contiguous()
    assert selected.value.is_contiguous()
    assert entry.select_heads(4) is entry
