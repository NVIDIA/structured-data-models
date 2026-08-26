from typing import cast

import torch

from sdm.cache import Cache, KVCacheEntry, KVCacheOffload
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
def test_cache_new_empty(device: torch.device) -> None:
    cache = Cache(existing="value", kv_cache_offload="layer").freeze()

    empty = cache.new_empty()

    assert type(empty) is type(cache)
    assert len(empty) == 0
    assert empty.is_recording

    empty["entry"] = KVCacheEntry(
        key=torch.ones(1, device=device),
        value=torch.ones(1, device=device),
    )
    entry = cast(KVCacheEntry, empty["entry"])
    if device.type == "cuda":
        assert entry.key.is_cpu
        assert entry.key.is_pinned()
    else:
        assert entry.key.device == device


@withCUDA
def test_layer_kv_cache_offload(device: torch.device) -> None:
    cache = Cache(kv_cache_offload="layer")
    other = torch.arange(5, device=device, dtype=torch.float32)
    cache["entry"] = KVCacheEntry(
        key=torch.arange(5, device=device, dtype=torch.float32),
        value=torch.arange(5, device=device, dtype=torch.float32),
    )
    cache["other"] = other

    entry = cast(KVCacheEntry, cache["entry"])
    assert cache["other"] is other
    if device.type == "cuda":
        assert entry.key.is_cpu
        assert entry.value.is_cpu
        assert entry.key.is_pinned()
        assert entry.value.is_pinned()
    else:
        assert entry.key.device == device
        assert entry.value.device == device

    moved = cache.to(device, non_blocking=True)
    moved["next"] = KVCacheEntry(
        key=torch.arange(5, device=device, dtype=torch.float32),
        value=torch.arange(5, device=device, dtype=torch.float32),
    )
    next_entry = cast(KVCacheEntry, moved["next"])
    if device.type == "cuda":
        assert next_entry.key.is_cpu
        assert next_entry.value.is_cpu
        assert next_entry.key.is_pinned()
        assert next_entry.value.is_pinned()
    else:
        assert next_entry.key.device == device
        assert next_entry.value.device == device


@onlyCUDA
def test_layer_kv_cache_offload_reduces_peak_cuda_memory() -> None:
    device = torch.device("cuda")

    def record(offload: KVCacheOffload) -> tuple[Cache, int]:
        cache = Cache(kv_cache_offload=offload)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        for i in range(4):
            cache[i] = KVCacheEntry(
                key=torch.empty(1 << 20, device=device),
                value=torch.empty(1 << 20, device=device),
            )
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline
        return cache, peak

    layer_cache, layer_peak = record("layer")
    assert all(tensor.is_cpu for tensor in layer_cache._tensors())
    assert all(tensor.is_pinned() for tensor in layer_cache._tensors())

    retained_cache, retained_peak = record("none")
    assert all(tensor.is_cuda for tensor in retained_cache._tensors())
    assert layer_peak < retained_peak
