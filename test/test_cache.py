# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

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


def test_cache_stack() -> None:
    caches = [
        Cache(
            entry=KVCacheEntry(key=torch.randn(3, 2), value=torch.randn(3, 4)),
            mean=torch.randn(1, 2),
        ).freeze()
        for _ in range(2)
    ]

    stacked = Cache.stack(caches)

    assert stacked.is_replaying
    assert stacked.keys() == caches[0].keys()
    entry = cast(KVCacheEntry, stacked["entry"])
    assert entry.key.size() == (2, 3, 2)
    assert entry.value.size() == (2, 3, 4)
    assert cast(torch.Tensor, stacked["mean"]).size() == (2, 1, 2)
    for i, cache in enumerate(caches):
        assert torch.equal(
            entry.key[i], cast(KVCacheEntry, cache["entry"]).key
        )
        assert torch.equal(
            entry.value[i], cast(KVCacheEntry, cache["entry"]).value
        )
        assert torch.equal(
            cast(torch.Tensor, stacked["mean"])[i],
            cast(torch.Tensor, cache["mean"]),
        )

    with pytest.raises(ValueError, match="share shapes"):
        Cache.stack(
            [Cache(mean=torch.randn(1, 2)), Cache(mean=torch.randn(2))]
        )
    with pytest.raises(ValueError, match="must be tensors"):
        Cache.stack([Cache(trees=[1]), Cache(trees=[2])])
