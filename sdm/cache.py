"""Cache primitives."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import NamedTuple

import torch
from torch import Tensor


class KVCacheEntry(NamedTuple):
    r"""Cached key/value projections for a single transformer block.

    Args:
        key: Cached key projection tensor.
        value: Cached value projection tensor.
    """

    key: Tensor
    value: Tensor

    def to(self, device: torch.device | str | None) -> "KVCacheEntry":
        r"""Perform :class:`~torch.Tensor` device conversion.

        Args:
            device: The device.
        """
        return self.__class__(
            key=self.key.to(device),
            value=self.value.to(device),
        )

    def cpu(self) -> "KVCacheEntry":
        r"""Copy :class:`KVCacheEntry` data in CPU memory."""
        return self.to("cpu")

    def cuda(
        self,
        device: torch.device | str | int | None = None,
    ) -> "KVCacheEntry":
        r"""Copy :class:`KVCacheEntry` data in CUDA memory."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(torch.device("cuda", device))
        return self.to(device)


class Cache(MutableMapping[str, object]):
    r"""A mutable mapping of model cache values."""

    def __init__(
        self,
        *args: Mapping[str, object] | Iterable[tuple[str, object]],
        **kwargs: object,
    ) -> None:
        self._items: dict[str, object] = dict(*args, **kwargs)

    def __getitem__(self, key: str) -> object:
        return self._items[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(self._items)

    def to(self, device: torch.device | str | None) -> "Cache":
        r"""Perform nested :class:`~torch.Tensor` device conversion.

        Args:
            device: The device.
        """

        def _to(value: object, device: torch.device | str | None) -> object:
            if isinstance(value, Tensor):
                return value.to(device)
            if isinstance(value, KVCacheEntry):
                return value.to(device)
            if isinstance(value, Cache):
                return value.to(device)
            if isinstance(value, list):
                return [_to(item, device=device) for item in value]
            if isinstance(value, tuple):
                return tuple(_to(item, device=device) for item in value)
            if isinstance(value, dict):
                return {
                    key: _to(item, device=device)
                    for key, item in value.items()
                }
            return value

        return self.__class__({k: _to(v, device) for k, v in self.items()})

    def cpu(self) -> "Cache":
        r"""Copy :class:`Cache` data in CPU memory."""
        return self.to("cpu")

    def cuda(
        self,
        device: torch.device | str | int | None = None,
    ) -> "Cache":
        r"""Copy :class:`Cache` data in CUDA memory."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(torch.device("cuda", device))
        return self.to(device)
