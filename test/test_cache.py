# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import pickle
from typing import cast

import pytest
import torch

from sdm.cache import Cache, KVCacheEntry
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
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("serialization", ["pickle", "torch", "deepcopy"])
def test_cache_serialization(
    device: torch.device,
    frozen: bool,
    serialization: str,
) -> None:
    pinned = (
        torch.arange(3).pin_memory()
        if device.type == "cuda"
        else torch.arange(3)
    )
    cache = Cache(
        entry=KVCacheEntry(key=pinned, value=torch.arange(4)),
        nested=[
            Cache(values={"pair": (pinned, torch.ones(2, device=device))})
        ],
        empty=Cache(),
        label="context",
    )
    if frozen:
        cache.freeze()
    assert copy.copy(cache)["entry"] is cache["entry"]

    if serialization == "pickle":
        restored = pickle.loads(pickle.dumps(cache))
    elif serialization == "deepcopy":
        restored = copy.deepcopy(cache)
    else:
        stream = io.BytesIO()
        torch.save(cache, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)

    entry = cast(KVCacheEntry, restored["entry"])
    torch.testing.assert_close(entry.key, pinned)
    torch.testing.assert_close(entry.value, torch.arange(4))
    assert entry.key.is_pinned() == (device.type == "cuda")
    assert not entry.value.is_pinned()
    nested = cast(list[Cache], restored["nested"])[0]
    pair = cast(
        dict[str, tuple[torch.Tensor, torch.Tensor]], nested["values"]
    )["pair"]
    torch.testing.assert_close(pair[0], pinned)
    torch.testing.assert_close(pair[1], torch.ones(2, device=device))
    assert pair[0].is_pinned() == (device.type == "cuda")
    assert pair[1].device == device
    assert not pair[1].is_pinned()
    assert restored["label"] == "context"
    assert len(cast(Cache, restored["empty"])) == 0
    for item in (restored, nested, cast(Cache, restored["empty"])):
        assert item.is_replaying == frozen
        if frozen:
            with pytest.raises(RuntimeError, match="record"):
                item["new"] = 1
        else:
            item["new"] = 1
            assert item["new"] == 1


@onlyCUDA
@pytest.mark.parametrize(
    ("map_location", "cuda_available"),
    [("cpu", False), ("cpu", True), ("cuda:0", True)],
)
def test_cache_serialization_device_mapping(
    map_location: str,
    cuda_available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache(
        pinned=torch.arange(3).pin_memory(),
        pageable=torch.arange(4),
    ).freeze()
    stream = io.BytesIO()
    torch.save(cache, stream)
    stream.seek(0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: cuda_available)
        restored = torch.load(
            stream,
            map_location=map_location,
            weights_only=False,
        )
    pinned = cast(torch.Tensor, restored["pinned"])
    pageable = cast(torch.Tensor, restored["pageable"])
    torch.testing.assert_close(pinned, torch.arange(3, device=map_location))
    torch.testing.assert_close(pageable, torch.arange(4, device=map_location))
    assert pinned.is_pinned() == (cuda_available and map_location == "cpu")
    assert not pageable.is_pinned()
    assert restored.is_replaying


def test_cache_serialization_without_pinning_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache(value=torch.arange(3)).freeze()
    with monkeypatch.context() as patch:
        patch.setattr(Cache, "__getstate__", lambda self: self.__dict__.copy())
        serialized = pickle.dumps(cache)
    restored = pickle.loads(serialized)
    torch.testing.assert_close(restored["value"], torch.arange(3))
    assert not restored["value"].is_pinned()
    assert restored.is_replaying
