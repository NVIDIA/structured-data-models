"""Generic runtime cache primitives for SchemaFM models."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import torch
from torch import Tensor

CacheKey = Hashable
T = TypeVar("T")


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Cache names must be non-empty strings")


def _validate_key(key: CacheKey) -> None:
    if key is None:
        raise ValueError("Cache group keys cannot be None")
    if not isinstance(key, Hashable):
        raise TypeError("Cache group keys must be hashable")


def _validate_value(value: object) -> None:
    if value is None:
        raise ValueError("Cache values cannot be None")


@dataclass(frozen=True)
class KVCacheEntry:
    r"""Cached key/value projections for one attention site.

    The entry intentionally does not know where it is stored or whether it is
    valid for a particular request. The owning model or caller manages cache
    keys and invalidation.
    """

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        r"""Validate tensors."""
        if not isinstance(self.key, Tensor):
            raise TypeError("KVCacheEntry.key must be a torch.Tensor")
        if not isinstance(self.value, Tensor):
            raise TypeError("KVCacheEntry.value must be a torch.Tensor")

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCacheEntry:
        r"""Move/cast cached tensors and return a new entry."""
        if device is None and dtype is None:
            return self
        if device is None:
            assert dtype is not None
            key = self.key.to(dtype=dtype)
            value = self.value.to(dtype=dtype)
        elif dtype is None:
            key = self.key.to(device=device)
            value = self.value.to(device=device)
        else:
            key = self.key.to(device=device, dtype=dtype)
            value = self.value.to(device=device, dtype=dtype)
        return KVCacheEntry(
            key=key,
            value=value,
        )


class CacheGroup(Generic[T]):
    r"""Named group of related cache values inside :class:`ModelCache`.

    Groups are generic. They can store per-layer :class:`KVCacheEntry` values,
    per-slice model artifacts, or any other repeated cache values a model owns.
    """

    def __init__(
        self,
        name: str,
        value_type: type[T] | None = None,
    ) -> None:
        r"""Initialize an empty cache group."""
        _validate_name(name)
        self.name = name
        self._value_type = value_type
        self._items: dict[CacheKey, T] = {}

    def _validate_group_value(self, value: object) -> T:
        _validate_value(value)
        if self._value_type is not None and not isinstance(
            value,
            self._value_type,
        ):
            raise TypeError(
                f"Cache group {self.name!r} expects "
                f"{self._value_type.__name__}; got {type(value).__name__}"
            )
        return cast(T, value)

    def __setitem__(self, key: CacheKey, value: T) -> None:
        r"""Store a value for a cache key."""
        _validate_key(key)
        self._items[key] = self._validate_group_value(value)

    def __getitem__(self, key: CacheKey) -> T:
        r"""Return a cached value, or raise when the key is absent."""
        _validate_key(key)
        return self._items[key]

    def get(self, key: CacheKey) -> T | None:
        r"""Return a cached value, or ``None`` when the key is absent."""
        _validate_key(key)
        return self._items.get(key)

    def __delitem__(self, key: CacheKey) -> None:
        r"""Remove a cached value."""
        _validate_key(key)
        del self._items[key]

    def clear(self) -> None:
        r"""Clear all cached values in this group."""
        self._items.clear()

    def __contains__(self, key: object) -> bool:
        r"""Return whether the group contains a cache key."""
        return key in self._items

    def __len__(self) -> int:
        r"""Return the number of items in this group."""
        return len(self._items)


class ModelCache:
    r"""Model-owned runtime cache for named artifacts and groups.

    ``ModelCache`` owns model-level cache lifecycle and invalidation. Attention
    modules should not receive this object; they should only return or consume
    values such as :class:`KVCacheEntry`.
    """

    def __init__(self) -> None:
        r"""Initialize an empty model cache."""
        self._items: dict[str, object] = {}

    def create_group(
        self,
        name: str,
        value_type: type[T] | None = None,
    ) -> CacheGroup[T]:
        r"""Create and return a named group."""
        _validate_name(name)
        if name in self._items:
            raise KeyError(f"Duplicate cache name {name!r}")
        group: CacheGroup[T] = CacheGroup(name, value_type=value_type)
        self._items[name] = group
        return group

    def group(
        self,
        name: str,
        value_type: type[T] | None = None,
    ) -> CacheGroup[T]:
        r"""Return a named group, or raise when absent."""
        value = self[name]
        if not isinstance(value, CacheGroup):
            raise TypeError(
                f"Cache item {name!r} is {type(value).__name__}; "
                "expected CacheGroup"
            )
        if value_type is not None and value._value_type is not value_type:
            stored = (
                "untyped values"
                if value._value_type is None
                else value._value_type.__name__
            )
            raise TypeError(
                f"Cache group {name!r} stores {stored}; "
                f"expected {value_type.__name__}"
            )
        return cast(CacheGroup[T], value)

    def get_group(
        self,
        name: str,
        value_type: type[T] | None = None,
    ) -> CacheGroup[T] | None:
        r"""Return a named group, or ``None`` when absent."""
        _validate_name(name)
        if name not in self._items:
            return None
        return self.group(name, value_type=value_type)

    def __setitem__(self, name: str, value: object) -> None:
        r"""Store an artifact by name."""
        _validate_name(name)
        _validate_value(value)
        if isinstance(self._items.get(name), CacheGroup):
            raise KeyError(f"Cache name {name!r} already exists as a group")
        self._items[name] = value

    def __getitem__(self, name: str) -> object:
        r"""Return an artifact or group by name, or raise when absent."""
        _validate_name(name)
        return self._items[name]

    def get(self, name: str, typ: type[T] | None = None) -> object | T | None:
        r"""Return an artifact or group, or ``None`` when absent."""
        _validate_name(name)
        value = self._items.get(name)
        if value is None or typ is None:
            return value
        if not isinstance(value, typ):
            raise TypeError(
                f"Cache item {name!r} has type "
                f"{type(value).__name__}; expected {typ.__name__}"
            )
        return value

    def __delitem__(self, name: str) -> None:
        r"""Remove an artifact or group by name."""
        _validate_name(name)
        del self._items[name]

    def clear(self) -> None:
        r"""Clear all artifacts and groups."""
        self._items.clear()

    def keys(self):
        r"""Return stored artifact and group names."""
        return self._items.keys()

    def __contains__(self, name: object) -> bool:
        r"""Return whether the cache contains a name."""
        return name in self._items

    def __len__(self) -> int:
        r"""Return the number of artifacts and groups in this cache."""
        return len(self._items)
