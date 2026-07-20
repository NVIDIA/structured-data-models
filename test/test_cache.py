from typing import cast

import pytest
import torch
from sdm.cache import Cache, KVCacheEntry


def test_cache() -> None:
    cache = Cache(foo="foo")
    assert len(cache) == 1
    cache["entry"] = KVCacheEntry(key=torch.randn(5), value=torch.randn(5))
    assert len(cache) == 2
    cache = cache.cpu()
    assert cast(KVCacheEntry, cache["entry"]).key.is_cpu
    assert cast(KVCacheEntry, cache["entry"]).value.is_cpu


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
        metadata=torch.ones(100),
    )

    assert cache.size == 3 * 4 + 2 * 8 + 5 + 4 * 2


def test_freeze_nested_caches() -> None:
    nested_caches = [Cache(), Cache(), Cache(), Cache()]
    cache = Cache(
        direct=nested_caches[0],
        list=[nested_caches[1]],
        tuple=(nested_caches[2],),
        dict={"cache": nested_caches[3]},
    )

    assert cache.freeze() is cache
    for nested_cache in [cache, *nested_caches]:
        assert nested_cache.is_replaying
        with pytest.raises(RuntimeError, match="requires the cache"):
            nested_cache["value"] = 1


def test_nested_cache_device_ignores_empty_children() -> None:
    cache = Cache(empty=Cache(), nested={"value": torch.ones(1)})

    assert cache.device == torch.device("cpu")
