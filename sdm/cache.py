"""Cache primitives."""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from enum import Enum
from typing import NamedTuple

import torch
from torch import Tensor
from typing_extensions import Self

from sdm.tensor.mixin import DeviceMixin


class _KVCacheEntry(NamedTuple):
    key: Tensor
    value: Tensor


class KVCacheEntry(_KVCacheEntry, DeviceMixin):
    r"""Cached key/value projections for a single transformer block.

    Args:
        key: Cached key projection tensor.
        value: Cached value projection tensor.
    """

    def to(self, device: torch.device | str | None) -> Self:
        return self.__class__(
            key=self.key.to(device),
            value=self.value.to(device),
        )

    @property
    def device(self) -> torch.device:
        return self.key.device


class Cache(MutableMapping[str, object], DeviceMixin):
    r"""A mutable mapping of model cache values."""

    class Mode(str, Enum):
        r"""The operating mode of a :class:`Cache`.

        A cache alternates between two phases: (1) recording key/value
        projections from a fit pass, and (2) replaying them across subsequent
        predict passes.
        Possible values are:

        Attributes:
            record: Allows recording of data.
            replay: Allows replaying of data.
        """

        record = "record"
        replay = "replay"

    def __init__(
        self,
        *args: Mapping[str, object] | Iterable[tuple[str, object]],
        **kwargs: object,
    ) -> None:
        self._mode = Cache.Mode.record
        self._items: dict[str, object] = dict(*args, **kwargs)

    @property
    def is_recording(self) -> bool:
        r"""Whether the cache is in recording mode."""
        return self._mode == Cache.Mode.record

    @property
    def is_replaying(self) -> bool:
        r"""Whether the cache is in replaying mode."""
        return self._mode == Cache.Mode.replay

    def freeze(self) -> None:
        r"""Freeze the cache to replay mode."""
        self._mode = Cache.Mode.replay

    def __setitem__(self, key: str, value: object) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__setitem__' requires the cache to be in 'record' mode"
            )
        self._items[key] = value

    def __delitem__(self, key: str) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__delitem__' requires the cache to be in 'record' mode"
            )
        del self._items[key]

    def __getitem__(self, key: str) -> object:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(self._items)

    def to(self, device: torch.device | str | None) -> Self:
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

        out = self.__class__({k: _to(v, device) for k, v in self.items()})
        out._mode = self._mode
        return out
