"""Cache primitives."""

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from enum import StrEnum
from typing import NamedTuple, Self

import torch
from torch import Tensor

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

    def to(
        self,
        device: torch.device | str | None,
        *,
        non_blocking: bool = False,
    ) -> Self:
        r""":meta private:"""  # noqa: D415
        return self.__class__(
            key=self.key.to(device, non_blocking=non_blocking),
            value=self.value.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Self:
        r"""Copy cached tensors into pinned CPU memory."""
        return self.__class__(
            key=self.key.pin_memory(),
            value=self.value.pin_memory(),
        )

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415
        devices = {self.key.device, self.value.device}
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected key and value cache tensors to be on the same "
                f"device (got '{self.key.device}' and '{self.value.device}')"
            )
        return next(iter(devices))


class Cache(MutableMapping[Hashable, object], DeviceMixin):
    r"""A mutable mapping of model cache values."""

    class Mode(StrEnum):
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
        *args: Mapping[Hashable, object] | Iterable[tuple[Hashable, object]],
        **kwargs: object,
    ) -> None:
        self._mode = Cache.Mode.record
        self._items: dict[Hashable, object] = dict(*args, **kwargs)

    @property
    def is_recording(self) -> bool:
        r"""Whether the cache is in recording mode."""
        return self._mode == Cache.Mode.record

    @property
    def is_replaying(self) -> bool:
        r"""Whether the cache is in replaying mode."""
        return self._mode == Cache.Mode.replay

    def size(self) -> int:
        r"""The size in bytes of key/value entries in supported containers."""

        def _size(value: object) -> int:
            if isinstance(value, KVCacheEntry):
                return (
                    value.key.numel() * value.key.element_size()
                    + value.value.numel() * value.value.element_size()
                )
            if isinstance(value, list | tuple):
                return sum(_size(item) for item in value)
            if isinstance(value, dict | Cache):
                return sum(_size(item) for item in value.values())
            return 0

        return _size(self)

    def freeze(self) -> Self:
        r"""Freeze the cache to replay mode."""

        def _freeze(value: object) -> None:
            if isinstance(value, Cache):
                value._mode = Cache.Mode.replay
                for item in value.values():
                    _freeze(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    _freeze(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _freeze(item)

        _freeze(self)
        return self

    def __setitem__(self, key: Hashable, value: object) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__setitem__' requires the cache to be in 'record' mode"
            )
        self._items[key] = value

    def __delitem__(self, key: Hashable) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__delitem__' requires the cache to be in 'record' mode"
            )
        del self._items[key]

    def __getitem__(self, key: Hashable) -> object:
        return self._items[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(self._items)

    def to(
        self,
        device: torch.device | str | None,
        *,
        non_blocking: bool = False,
    ) -> Self:
        r""":meta private:"""  # noqa: D415

        def _to(value: object) -> object:
            if isinstance(value, Tensor):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, KVCacheEntry):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, Cache):
                return value.to(device, non_blocking=non_blocking)
            if isinstance(value, list):
                return [_to(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_to(item) for item in value)
            if isinstance(value, dict):
                return {key: _to(item) for key, item in value.items()}
            return value

        out = self.__class__({k: _to(v) for k, v in self.items()})
        out._mode = self._mode
        return out

    def pin_memory(self) -> Self:
        r"""Copy nested tensor data into pinned CPU memory."""

        def _pin_memory(value: object) -> object:
            if isinstance(value, Tensor):
                return value.pin_memory()
            if isinstance(value, KVCacheEntry):
                return value.pin_memory()
            if isinstance(value, Cache):
                return value.pin_memory()
            if isinstance(value, list):
                return [_pin_memory(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_pin_memory(item) for item in value)
            if isinstance(value, dict):
                return {key: _pin_memory(item) for key, item in value.items()}
            return value

        pinned = self.__class__(
            {key: _pin_memory(value) for key, value in self.items()}
        )
        pinned._mode = self._mode
        return pinned

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415

        def _devices(value: object) -> set[torch.device]:
            if isinstance(value, Tensor | KVCacheEntry | Cache):
                return {value.device}
            if isinstance(value, list | tuple):
                return {device for item in value for device in _devices(item)}
            if isinstance(value, dict):
                return {
                    device
                    for item in value.values()
                    for device in _devices(item)
                }
            return set()

        devices = {
            device for item in self.values() for device in _devices(item)
        }
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"{self.__class__.__name__!r}"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tensors in {self.__class__.__name__!r} to be on "
                f"the same device (got {list(devices)})"
            )
        return next(iter(devices))
