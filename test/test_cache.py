from typing import Any, cast

import pytest
import torch
from schemafm import CacheGroup, KVCacheEntry, ModelCache
from torch import Tensor


def _entry() -> KVCacheEntry:
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 5)
    return KVCacheEntry(
        key=key,
        value=value,
    )


def test_kv_cache_entry_to() -> None:
    entry = _entry()

    out = entry.to(dtype=torch.float64, non_blocking=True)

    assert out is not entry
    assert out.key.dtype == torch.float64
    assert out.value.dtype == torch.float64
    assert entry.to() is entry


def test_kv_cache_entry_validates_inputs() -> None:
    with pytest.raises(TypeError, match="key"):
        KVCacheEntry(
            key=cast(Tensor, object()),
            value=torch.randn(1),
        )

    with pytest.raises(TypeError, match="value"):
        KVCacheEntry(
            key=torch.randn(1),
            value=cast(Tensor, object()),
        )


def test_cache_group_dict_access() -> None:
    group = CacheGroup("icl.kv", value_type=KVCacheEntry)
    entry = _entry()

    group[0] = entry

    assert len(group) == 1
    assert 0 in group
    assert group[0] is entry
    assert group.get(0) is entry
    assert group.get(1) is None

    new_entry = _entry()
    group[0] = new_entry
    assert group[0] is new_entry

    del group[0]
    assert group.get(0) is None


def test_cache_group_errors() -> None:
    group = CacheGroup("icl.kv", value_type=KVCacheEntry)
    group[0] = _entry()

    with pytest.raises(KeyError):
        _ = group[1]

    with pytest.raises(ValueError, match="non-empty"):
        CacheGroup("")

    with pytest.raises(ValueError, match="None"):
        group[1] = cast(KVCacheEntry, None)

    with pytest.raises(ValueError, match="keys cannot be None"):
        group[cast(Any, None)] = _entry()

    with pytest.raises(ValueError, match="keys cannot be None"):
        group.get(cast(Any, None))

    with pytest.raises(TypeError, match="expects KVCacheEntry"):
        group[1] = cast(KVCacheEntry, object())


def test_cache_group_clear() -> None:
    group = CacheGroup("icl.kv", value_type=KVCacheEntry)
    group[0] = _entry()
    group[1] = _entry()

    group.clear()

    assert len(group) == 0


def test_model_cache_artifacts() -> None:
    cache = ModelCache()
    artifact = {"mean": torch.zeros(2)}

    cache["preprocessor"] = artifact

    assert len(cache) == 1
    assert "preprocessor" in cache
    assert list(cache.keys()) == ["preprocessor"]
    assert cache["preprocessor"] is artifact
    assert cache.get("preprocessor", dict) is artifact
    assert cache.get("missing", dict) is None

    replacement = {"mean": torch.ones(2)}
    cache["preprocessor"] = replacement
    assert cache["preprocessor"] is replacement


def test_model_cache_groups() -> None:
    cache = ModelCache()
    group = cache.create_group("icl.kv", value_type=KVCacheEntry)

    group[0] = _entry()

    assert cache.group("icl.kv", value_type=KVCacheEntry) is group
    assert cache.get_group("icl.kv", value_type=KVCacheEntry) is group
    assert list(cache.keys()) == ["icl.kv"]
    assert "icl.kv" in cache
    assert len(cache) == 1
    assert group[0].key.size() == (2, 3, 4, 5)


def test_model_cache_group_lookup_does_not_create() -> None:
    cache = ModelCache()

    assert cache.get_group("icl.kv") is None

    with pytest.raises(KeyError):
        cache.group("icl.kv")

    assert len(cache) == 0


def test_model_cache_namespace_collisions() -> None:
    cache = ModelCache()
    cache["preprocessor"] = object()

    with pytest.raises(KeyError, match="Duplicate"):
        cache.create_group("preprocessor")

    cache = ModelCache()
    cache.create_group("icl.kv")

    with pytest.raises(KeyError, match="group"):
        cache["icl.kv"] = object()

    with pytest.raises(KeyError, match="Duplicate"):
        cache.create_group("icl.kv")


def test_model_cache_group_type_errors() -> None:
    cache = ModelCache()
    cache.create_group("icl.kv", value_type=KVCacheEntry)

    with pytest.raises(TypeError, match="stores KVCacheEntry"):
        cache.get_group("icl.kv", value_type=torch.Tensor)

    with pytest.raises(TypeError, match="stores KVCacheEntry"):
        cache.group("icl.kv", value_type=torch.Tensor)

    untyped_cache = ModelCache()
    untyped_cache.create_group("untyped")

    with pytest.raises(TypeError, match="stores untyped values"):
        untyped_cache.group("untyped", value_type=KVCacheEntry)

    cache["artifact"] = object()

    with pytest.raises(TypeError, match="expected CacheGroup"):
        cache.group("artifact")


def test_model_cache_type_and_missing_errors() -> None:
    cache = ModelCache()
    cache["preprocessor"] = object()

    with pytest.raises(TypeError, match="expected dict"):
        cache.get("preprocessor", dict)

    with pytest.raises(KeyError):
        _ = cache["missing"]

    with pytest.raises(ValueError, match="non-empty"):
        cache[""] = object()

    with pytest.raises(ValueError, match="non-empty"):
        cache.get("", object)

    with pytest.raises(ValueError, match="non-empty"):
        cache.get_group("")

    with pytest.raises(ValueError, match="None"):
        cache["none"] = None


def test_model_cache_clear_and_delete() -> None:
    cache = ModelCache()
    cache["preprocessor"] = object()
    cache.create_group("icl.kv", value_type=KVCacheEntry)[0] = _entry()

    del cache["preprocessor"]
    assert "preprocessor" not in cache
    assert "icl.kv" in cache

    del cache["icl.kv"]
    assert len(cache) == 0

    cache["preprocessor"] = object()
    cache.create_group("icl.kv", value_type=KVCacheEntry)[0] = _entry()
    cache.clear()
    assert len(cache) == 0
