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


def test_cache_freeze_recursively() -> None:
    children = (Cache(), Cache(), Cache())
    cache = Cache(nested=[children[0], (children[1], {"cache": children[2]})])

    cache.freeze()

    assert cache.is_replaying
    assert all(child.is_replaying for child in children)


def test_cache_to_preserves_modes_and_source() -> None:
    source_tensors = [torch.arange(2), torch.arange(2, 4), torch.arange(4, 6)]
    snapshots = [tensor.clone() for tensor in source_tensors]
    recording = Cache(tensor=source_tensors[1])
    replaying = Cache(tensor=source_tensors[2])
    replaying.freeze()
    source = Cache(
        tensor=source_tensors[0],
        children=[recording, {"replaying": replaying}],
    )

    converted = source.to("meta")
    converted_children = cast(list[object], converted["children"])
    converted_recording = cast(Cache, converted_children[0])
    converted_replaying = cast(
        Cache,
        cast(dict[str, object], converted_children[1])["replaying"],
    )
    converted_tensors = [
        cast(torch.Tensor, converted["tensor"]),
        cast(torch.Tensor, converted_recording["tensor"]),
        cast(torch.Tensor, converted_replaying["tensor"]),
    ]

    assert converted.is_recording
    assert converted_recording.is_recording
    assert converted_replaying.is_replaying
    assert all(tensor.device.type == "meta" for tensor in converted_tensors)
    assert source.is_recording
    assert recording.is_recording
    assert replaying.is_replaying
    assert all(
        tensor.is_cpu and torch.equal(tensor, snapshot)
        for tensor, snapshot in zip(source_tensors, snapshots)
    )

    converted["new"] = None
    converted_recording["new"] = None
    with pytest.raises(RuntimeError, match=r"requires.*'record' mode"):
        converted_replaying["new"] = None
    assert "new" not in source
    assert "new" not in recording
