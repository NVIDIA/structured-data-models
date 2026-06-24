"""Generic runtime cache primitives."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class KVCacheEntry:
    r"""Cached key/value projections for one attention site.

    The entry intentionally does not know where it is stored or whether it is
    valid for a particular request. The owning model or caller manages cache
    keys and invalidation. The key and value tensors stay separate to avoid
    baking a packed cache layout into this generic primitive.

    Args:
        key: Cached key projection tensor.
        value: Cached value projection tensor.
    """

    key: Tensor
    value: Tensor


class ModelCache:
    r"""Model-owned runtime cache for named values.

    ``ModelCache`` owns model-level cache lifecycle and invalidation. Attention
    modules should not receive this object; they should only return or consume
    values such as :class:`KVCacheEntry`. Repeated values, such as per-layer KV
    entries, can be stored as plain dictionaries.
    """

    def __init__(self) -> None:
        r"""Initialize an empty model cache."""
        self._items: dict[str, object] = {}

    def __setitem__(self, name: str, value: object) -> None:
        r"""Store a value by name."""
        self._items[name] = value

    def __getitem__(self, name: str) -> object:
        r"""Return a value by name, or raise when absent."""
        return self._items[name]

    def get(self, name: str, default: object = None) -> object:
        r"""Return a value by name, or ``default`` when absent."""
        return self._items.get(name, default)

    def __delitem__(self, name: str) -> None:
        r"""Remove a value by name."""
        del self._items[name]

    def clear(self) -> None:
        r"""Clear all cached values."""
        self._items.clear()

    def keys(self):
        r"""Return stored value names."""
        return self._items.keys()

    def __contains__(self, name: object) -> bool:
        r"""Return whether the cache contains a name."""
        return name in self._items

    def __len__(self) -> int:
        r"""Return the number of values in this cache."""
        return len(self._items)
