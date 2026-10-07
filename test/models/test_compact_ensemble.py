# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from research.multigpu.compact_ensemble import compact_cache, storage_bytes

from sdm.cache import Cache, KVCacheEntry


def test_compact_unused_storage_and_preserve_aliases() -> None:
    projection = torch.arange(120.0).view(10, 12)
    value = projection[:, 6:]
    key = value.square()
    cache = Cache(kv=KVCacheEntry(key, value), same_value=value).freeze()
    compact = compact_cache(cache)
    assert compact.is_replaying
    assert storage_bytes(cache) == 720
    assert storage_bytes(compact) == 480
    original = list(cache._tensors())
    actual = list(compact._tensors())
    for before, after in zip(original, actual, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert actual[1].data_ptr() == actual[2].data_ptr()


def test_fully_referenced_shared_storage_is_not_copied() -> None:
    projection = torch.arange(120.0).view(10, 12)
    key, value = projection.chunk(2, dim=-1)
    cache = Cache(kv=KVCacheEntry(key, value)).freeze()
    compact = compact_cache(cache)
    assert storage_bytes(cache) == storage_bytes(compact) == 480
    for before, after in zip(
        cache._tensors(), compact._tensors(), strict=True
    ):
        assert before.data_ptr() == after.data_ptr()
