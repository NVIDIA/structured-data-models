# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm.cache import Cache, KVCacheEntry, QuantizedKVCacheEntry
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
