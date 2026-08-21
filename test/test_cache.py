import pickle
from typing import cast

import pytest
import torch

from sdm import TableTensor
from sdm.cache import Cache, CacheManager, KVCacheEntry
from sdm.testing import onlyCUDA, withCUDA


def _raise_cache_load_error(
    manager: CacheManager,
    caches: tuple[Cache, ...],
    device: torch.device,
) -> None:
    with manager.load(caches, device) as loaded_caches:
        cast(torch.Tensor, next(loaded_caches)["value"]).sum()
        raise RuntimeError("model failure")


def _consume_caches(
    manager: CacheManager,
    caches: tuple[Cache, ...],
    device: torch.device,
) -> None:
    with manager.load(caches, device) as loaded_caches:
        for cache in loaded_caches:
            cast(torch.Tensor, cache["value"]).sum()


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
def test_cache_manager_loads_different_tensor_shapes(
    device: torch.device,
) -> None:
    caches = (
        Cache(
            value=torch.arange(5, dtype=torch.float32),
            entry=KVCacheEntry(
                key=torch.arange(3, dtype=torch.int64),
                value=torch.tensor(True),
            ),
        ).freeze(),
        Cache(
            value=torch.arange(9, dtype=torch.float16),
            entry=KVCacheEntry(
                key=torch.arange(7, dtype=torch.int32),
                value=torch.tensor(False),
            ),
        ).freeze(),
        Cache(
            value=torch.arange(4, dtype=torch.float64),
            entry=KVCacheEntry(
                key=torch.arange(11, dtype=torch.int16),
                value=torch.tensor(True),
            ),
        ).freeze(),
        Cache(
            value=torch.arange(13, dtype=torch.bfloat16),
            entry=KVCacheEntry(
                key=torch.arange(2, dtype=torch.int8),
                value=torch.tensor(False),
            ),
        ).freeze(),
    )
    if device.type == "cuda":
        caches = tuple(cache.pin_memory() for cache in caches)
    manager = CacheManager()

    for _ in range(2):
        with manager.load(caches, device) as loaded_caches:
            for cache, copied in zip(caches, loaded_caches, strict=True):
                assert copied.is_replaying
                for source, target in zip(
                    cache._tensors(),
                    copied._tensors(),
                    strict=True,
                ):
                    torch.testing.assert_close(target.cpu(), source)


@withCUDA
def test_cache_manager_preserves_non_packable_tensors(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(torch.arange(4.0).view(2, 2))
    sparse = torch.arange(4.0).view(2, 2).to_sparse()
    caches = (
        Cache(table=table, sparse=sparse).freeze(),
        Cache(
            table=TableTensor.from_tensor(torch.arange(1.0, 5.0).view(2, 2)),
            sparse=torch.arange(1.0, 5.0).view(2, 2).to_sparse(),
        ).freeze(),
    )

    with CacheManager().load(caches, device) as loaded_caches:
        loaded = tuple(loaded_caches)

    for source, target in zip(caches, loaded, strict=True):
        source_table = cast(TableTensor, source["table"])
        target_table = cast(TableTensor, target["table"])
        source_sparse = cast(torch.Tensor, source["sparse"])
        target_sparse = cast(torch.Tensor, target["sparse"])

        assert isinstance(target_table, TableTensor)
        assert target_sparse.layout == torch.sparse_coo
        assert target_table.device == device
        assert target_sparse.device == device
        torch.testing.assert_close(
            target_table.numerical.cpu(),
            source_table.numerical,
        )
        torch.testing.assert_close(target_sparse.cpu(), source_sparse)


@withCUDA
def test_cache_manager_reusable_after_error(device: torch.device) -> None:
    caches = tuple(
        Cache(value=torch.full((8,), float(i))).freeze() for i in range(3)
    )
    if device.type == "cuda":
        caches = tuple(cache.pin_memory() for cache in caches)
    manager = CacheManager()

    with pytest.raises(RuntimeError, match="model failure"):
        _raise_cache_load_error(manager, caches, device)

    restored = pickle.loads(pickle.dumps(manager))
    results: list[torch.Tensor] = []
    with restored.load(caches, device) as loaded_caches:
        for cache in loaded_caches:
            results.append(cast(torch.Tensor, cache["value"]).sum())

    torch.testing.assert_close(
        torch.stack(results).cpu(),
        torch.tensor([0.0, 8.0, 16.0]),
    )


@withCUDA
def test_cache_manager_stores_ensemble_caches(
    device: torch.device,
) -> None:
    manager = CacheManager()
    cache = Cache(value=torch.arange(5, device=device)).freeze()

    single = manager.store(
        cache,
        num_estimators=1,
        device=device,
    )
    ensemble = manager.store(
        cache,
        num_estimators=2,
        device=device,
    )

    assert single is cache
    if device.type == "cuda":
        assert ensemble.is_cpu
        assert ensemble.is_pinned()
    else:
        assert ensemble is cache


@onlyCUDA
def test_cache_manager_bounds_and_releases_staging_memory() -> None:
    device = torch.device("cuda")
    caches = tuple(
        Cache(value=torch.empty(1 << 20)).pin_memory().freeze()
        for _ in range(3)
    )
    cache_size = caches[0].size()
    manager = CacheManager()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)

    _consume_caches(manager, caches, device)
    torch.cuda.synchronize(device)

    assert torch.cuda.memory_allocated(device) - baseline == 2 * cache_size

    _consume_caches(manager, caches, torch.device("cpu"))
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == baseline

    _consume_caches(manager, caches, device)
    manager.clear()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_allocated(device) == baseline
