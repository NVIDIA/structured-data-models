from typing import cast

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


def test_cache_freeze_recursively() -> None:
    children = (Cache(), Cache(), Cache())
    cache = Cache(nested=[children[0], (children[1], {"cache": children[2]})])

    cache.freeze()

    assert cache.is_replaying
    assert all(child.is_replaying for child in children)


def test_cache_freeze_handles_cycles_and_shared_values() -> None:
    child = Cache()
    cycle: list[object] = [child]
    cycle.append(cycle)
    cache = Cache(child=child, alias={"child": child}, cycle=cycle)
    child["parent"] = cache

    cache.freeze()

    assert cache.is_replaying
    assert child.is_replaying
