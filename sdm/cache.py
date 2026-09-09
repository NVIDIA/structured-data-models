from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from enum import StrEnum
from typing import Literal, NamedTuple, Self, TypeAlias

import torch
from torch import Tensor

from sdm.tensor.mixin import DeviceMixin

KVCacheOffload: TypeAlias = Literal["none", "estimator", "layer"]


class _KVCacheEntry(NamedTuple):
    key: Tensor
    value: Tensor


class KVCacheEntry(_KVCacheEntry, DeviceMixin):
    r"""Cached key/value projections for a single transformer block.

    Args:
        key: Cached key projection tensor.
        value: Cached value projection tensor.
    """

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(key=fn(self.key), value=fn(self.value))


class Cache(MutableMapping[Hashable, object], DeviceMixin):
    r"""A mutable mapping of model cache values.

    Args:
        args: Initial cache data as a mapping or iterable of key/value pairs.
        kv_cache_offload: Offloading policy for key/value entries. ``"none"``
            retains entries on their input device, ``"estimator"`` leaves
            entries unchanged for the owning model to offload together, and
            ``"layer"`` synchronously moves each entry to pinned CPU memory
            when recorded.
        kwargs: Additional initial cache data.
    """

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
        kv_cache_offload: KVCacheOffload = "none",
        **kwargs: object,
    ) -> None:
        self._mode = Cache.Mode.record
        self._kv_cache_offload = kv_cache_offload
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
        r"""The size in bytes of tensor data stored in this cache."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._tensors()
        )

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
            elif isinstance(value, Mapping):
                for item in value.values():
                    _freeze(item)

        _freeze(self)
        return self

    def __setitem__(self, key: Hashable, value: object) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__setitem__' requires the cache to be in 'record' mode"
            )
        if isinstance(value, KVCacheEntry):
            value = self._store_key_value(value)
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

    def _tensors(self) -> Iterator[Tensor]:
        def _iter_tensors(value: object) -> Iterator[Tensor]:
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, DeviceMixin):
                yield from value._tensors()
            elif isinstance(value, list | tuple):
                for item in value:
                    yield from _iter_tensors(item)
            elif isinstance(value, Mapping):
                for item in value.values():
                    yield from _iter_tensors(item)

        for value in self.values():
            yield from _iter_tensors(value)

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        def _apply(value: object) -> object:
            if isinstance(value, Tensor):
                return fn(value)
            if isinstance(value, DeviceMixin):
                return value._apply_tensor(fn)
            if isinstance(value, list):
                return [_apply(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_apply(item) for item in value)
            if isinstance(value, Mapping):
                return {key: _apply(item) for key, item in value.items()}
            return value

        out = self.new_empty()
        out._items.update({key: _apply(value) for key, value in self.items()})
        out._mode = self._mode
        return out

    def new_empty(self) -> Self:
        """Return an empty recording cache with the same configuration."""
        return self.__class__(kv_cache_offload=self._kv_cache_offload)

    def _offload(self, device: torch.device) -> Self:
        def _copy(tensor: Tensor) -> Tensor:
            if tensor.is_cuda:
                return _to_pinned_cpu(tensor, non_blocking=True)
            return tensor.pin_memory()

        try:
            return self._apply_tensor(_copy)
        finally:
            # Keep the source cache alive and finish host writes before return.
            torch.cuda.current_stream(device).synchronize()

    def _store_key_value(self, entry: KVCacheEntry) -> DeviceMixin:
        # Representation transforms, such as quantization, precede placement.
        stored: DeviceMixin = entry
        if self._kv_cache_offload == "layer" and any(
            tensor.is_cuda for tensor in stored._tensors()
        ):
            stored = stored._apply_tensor(_to_pinned_cpu)
        return stored


def _to_pinned_cpu(tensor: Tensor, *, non_blocking: bool = False) -> Tensor:
    if not tensor.is_cuda:
        return tensor
    out = torch.empty_like(tensor, device="cpu", pin_memory=True)
    return out.copy_(tensor, non_blocking=non_blocking)
