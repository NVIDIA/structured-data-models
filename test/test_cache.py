from typing import cast

import torch

from sdm import StringTensor
from sdm.cache import Cache, KVCacheEntry
from sdm.testing import onlyCUDA


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

    assert cache.size() == 3 * 4 + 2 * 8 + 5 + 4 * 2


@onlyCUDA
def test_cache_pinned_transfer() -> None:
    entry = KVCacheEntry(key=torch.randn(5), value=torch.randn(5))
    strings = StringTensor.from_list(["foo", "bar"])
    child = Cache(tensor=torch.randn(3))
    child.freeze()
    cache = Cache(entry=entry, strings=strings, nested=[child])
    cache.freeze()

    pinned = cache.pin_memory()
    assert isinstance(pinned["entry"], KVCacheEntry)
    assert isinstance(pinned["strings"], StringTensor)
    assert isinstance(pinned["nested"], list)
    pinned_child = pinned["nested"][0]
    assert isinstance(pinned_child, Cache)
    assert pinned.is_replaying
    assert pinned_child.is_replaying
    assert pinned["entry"].key.is_pinned()
    assert pinned["entry"].value.is_pinned()
    assert pinned["strings"].is_pinned()
    assert isinstance(pinned_child["tensor"], torch.Tensor)
    assert pinned_child["tensor"].is_pinned()

    staged = pinned.to("cuda", non_blocking=True)

    assert isinstance(staged["entry"], KVCacheEntry)
    assert isinstance(staged["strings"], StringTensor)
    torch.testing.assert_close(staged["entry"].key.cpu(), entry.key)
    torch.testing.assert_close(staged["entry"].value.cpu(), entry.value)
    assert staged["strings"].tolist() == strings.tolist()
    assert staged.is_replaying
