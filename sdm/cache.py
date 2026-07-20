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
        r""":meta private:"""  # noqa: D415
        return self.__class__(
            key=self.key.to(device),
            value=self.value.to(device),
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

    @property
    def size(self) -> int:
        r"""The size in bytes of key/value entries in supported containers."""

        def _size(value: object) -> int:
            if isinstance(value, KVCacheEntry):
                return (
                    value.key.numel() * value.key.element_size()
                    + value.value.numel() * value.value.element_size()
                )
            if isinstance(value, Cache):
                values = value.values()
            elif isinstance(value, list | tuple):
                values = value
            elif isinstance(value, dict):
                values = value.values()
            else:
                return 0
            return sum(_size(item) for item in values)

        return _size(self)

    def freeze(self) -> Self:
        r"""Freeze the cache to replay mode."""

        def _freeze(value: object) -> None:
            if isinstance(value, Cache):
                value.freeze()
            elif isinstance(value, list | tuple):
                for item in value:
                    _freeze(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _freeze(item)

        self._mode = Cache.Mode.replay
        for value in self.values():
            _freeze(value)
        return self

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
        r""":meta private:"""  # noqa: D415

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

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415

        def _devices(value: object) -> set[torch.device]:
            if isinstance(value, Tensor | KVCacheEntry):
                return {value.device}
            if isinstance(value, Cache):
                return {
                    device
                    for item in value.values()
                    for device in _devices(item)
                }
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
                f"'{self.__class__.__name__}'"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tensors in '{self.__class__.__name__}' to be on "
                f"the same device (got {list(devices)})"
            )
        return next(iter(devices))
