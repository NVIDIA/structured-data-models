import pytest
import torch
from schemafm.cache import KVCacheEntry, ModelCache


def _entry() -> KVCacheEntry:
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 5)
    return KVCacheEntry(
        key=key,
        value=value,
    )


def test_kv_cache_entry_stores_key_and_value() -> None:
    entry = _entry()

    assert entry.key.size() == (2, 3, 4, 5)
    assert entry.value.size() == (2, 3, 4, 5)


def test_model_cache_stores_named_values() -> None:
    cache = ModelCache()
    artifact = {"mean": torch.zeros(2)}

    cache["preprocessor"] = artifact

    assert len(cache) == 1
    assert "preprocessor" in cache
    assert list(cache.keys()) == ["preprocessor"]
    assert cache["preprocessor"] is artifact
    assert cache.get("preprocessor") is artifact
    assert cache.get("missing") is None

    replacement = {"mean": torch.ones(2)}
    cache["preprocessor"] = replacement
    assert cache["preprocessor"] is replacement


def test_model_cache_stores_plain_dict_groups() -> None:
    cache = ModelCache()
    kv_entries: dict[int, KVCacheEntry] = {}

    cache["icl.kv"] = kv_entries
    kv_entries[0] = _entry()

    assert cache["icl.kv"] is kv_entries
    assert kv_entries[0].key.size() == (2, 3, 4, 5)


def test_model_cache_delete_and_clear() -> None:
    cache = ModelCache()
    cache["preprocessor"] = object()
    cache["icl.kv"] = {0: _entry()}

    del cache["preprocessor"]
    assert "preprocessor" not in cache
    assert "icl.kv" in cache

    del cache["icl.kv"]
    assert len(cache) == 0

    cache["preprocessor"] = object()
    cache["icl.kv"] = {0: _entry()}
    cache.clear()
    assert len(cache) == 0


def test_model_cache_missing_key_raises() -> None:
    cache = ModelCache()

    with pytest.raises(KeyError):
        _ = cache["missing"]
