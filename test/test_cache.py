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


def test_cache_freeze_nested() -> None:
    nested_list = Cache()
    nested_dict = Cache()
    cache = Cache(
        nested_list=[nested_list],
        nested_dict={"cache": nested_dict},
    )

    cache.freeze()

    assert cache.is_replaying
    assert nested_list.is_replaying
    assert nested_dict.is_replaying
