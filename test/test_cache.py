from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.cache import Cache, KVCacheEntry
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


def test_cache_size_deduplicates_shared_storage() -> None:
    tensor = torch.ones(8)
    cache = Cache(
        tensor=tensor,
        view=tensor[2:],
        nested=[tensor],
    )

    assert cache.size() == tensor.untyped_storage().nbytes()


def test_cache_size_counts_custom_tensor_storage() -> None:
    categories = TableTensor.from_columns(
        {"value": ["a", "bc"]},
        stypes={"value": Stype.categorical},
    ).categorical.categories[0]
    cache = Cache(categories=categories)

    assert cache.size() == 3 + 3 * torch.tensor(0).element_size()
