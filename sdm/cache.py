from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from enum import StrEnum
from types import TracebackType
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

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(key=fn(self.key), value=fn(self.value))


def _iter_cache_tensors(value: object) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, DeviceMixin):
        yield from value._tensors()
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_cache_tensors(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_cache_tensors(item)


def _apply_cache_value(
    value: object,
    fn: Callable[[Tensor], Tensor],
) -> object:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, DeviceMixin):
        return value._apply_tensor(fn)
    if isinstance(value, list):
        return [_apply_cache_value(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_apply_cache_value(item, fn) for item in value)
    if isinstance(value, Mapping):
        return {
            key: _apply_cache_value(item, fn) for key, item in value.items()
        }
    return value


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
        for value in self.values():
            yield from _iter_cache_tensors(value)

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        out = self.__class__(
            {key: _apply_cache_value(value, fn) for key, value in self.items()}
        )
        out._mode = self._mode
        return out


class _CacheSlot:
    """Reusable device storage for one estimator cache."""

    # Preserve allocator-like alignment for typed views into the byte buffer.
    _alignment = 256

    def __init__(
        self,
        caches: Sequence[Cache],
        device: torch.device,
    ) -> None:
        size = max(self._packed_size(cache) for cache in caches)
        self._buffer = torch.empty(
            size,
            dtype=torch.uint8,
            device=device,
        )

    @classmethod
    def _align(cls, offset: int, element_size: int) -> int:
        alignment = max(cls._alignment, element_size)
        return (offset + alignment - 1) // alignment * alignment

    @classmethod
    def _packed_size(cls, cache: Cache) -> int:
        offset = 0
        for tensor in cache._tensors():
            if not cls._is_packable(tensor):
                continue
            offset = cls._align(offset, tensor.element_size())
            offset += tensor.numel() * tensor.element_size()
        return offset

    @staticmethod
    def _is_packable(tensor: Tensor) -> bool:
        # Tensor subclasses need their own conversion to preserve metadata.
        return (
            type(tensor) is Tensor
            and tensor.layout == torch.strided
            and not tensor.is_quantized
        )

    def copy_from(
        self,
        cache: Cache,
        *,
        non_blocking: bool = False,
    ) -> Cache:
        """Copy tensor data from ``cache`` into this slot."""
        offset = 0

        def _copy(tensor: Tensor) -> Tensor:
            nonlocal offset
            if not self._is_packable(tensor):
                return tensor.to(
                    self._buffer.device,
                    non_blocking=non_blocking,
                )

            element_size = tensor.element_size()
            offset = self._align(offset, element_size)
            size = tensor.numel() * element_size
            out = self._buffer.narrow(0, offset, size)
            out = out.view(tensor.dtype).view(tensor.size())
            out.copy_(tensor, non_blocking=non_blocking)
            offset += size
            return out

        return cache._apply_tensor(_copy)


class _CacheState:
    """Persistent CUDA staging state for one estimator ensemble."""

    def __init__(
        self,
        caches: Sequence[Cache],
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        self.caches = tuple(caches)
        self.device = device
        with torch.cuda.stream(stream):
            # Either slot can hold any estimator when the ensemble size is odd.
            self.slots: tuple[_CacheSlot, _CacheSlot] | None = (
                _CacheSlot(caches, device),
                _CacheSlot(caches, device),
            )
        self.next: tuple[Cache, torch.cuda.Event] | None = None
        self.next_slot = 0

    def matches(
        self,
        caches: Sequence[Cache],
        device: torch.device,
    ) -> bool:
        return (
            self.device == device
            and len(self.caches) == len(caches)
            and all(
                source is candidate
                for source, candidate in zip(
                    self.caches,
                    caches,
                    strict=True,
                )
            )
        )

    def dispose(self) -> None:
        self.next = None
        self.slots = None
        self.caches = ()


class _CacheLoader(Iterator[Cache]):
    """Context-managed traversal of caches staged by a manager."""

    def __init__(
        self,
        manager: "CacheManager",
        caches: Sequence[Cache],
        device: torch.device,
    ) -> None:
        self._manager = manager
        self._caches = tuple(caches)
        self._device = device
        self._index = 0
        self._current: Cache | None = None
        self._next: tuple[Cache, torch.cuda.Event] | None = None
        self._compute_stream: torch.cuda.Stream | None = None
        self._transfer_stream: torch.cuda.Stream | None = None
        self._state: _CacheState | None = None

    def __enter__(self) -> Self:
        if self._device.type != "cuda":
            self._manager._invalidate_for(self._device)
            return self

        try:
            self._compute_stream = torch.cuda.current_stream(self._device)
            self._transfer_stream = self._manager._stream(self._device)
            if len(self._caches) > 1:
                self._state = self._manager._acquire(
                    self._caches,
                    self._device,
                    self._transfer_stream,
                )
                self._next, self._state.next = self._state.next, None
            if self._next is None:
                self._next = self._stage(0)
        except BaseException:
            try:
                self._synchronize()
            finally:
                self._manager._finish(self._state, keep=False)
            raise
        return self

    def __next__(self) -> Cache:
        self._release()
        if self._index == len(self._caches):
            raise StopIteration

        if self._device.type != "cuda":
            cache = self._caches[self._index]
            self._index += 1
            return cache

        assert self._next is not None
        assert self._compute_stream is not None
        cache, ready = self._next
        self._compute_stream.wait_event(ready)

        self._index += 1
        if self._index < len(self._caches):
            self._next = self._stage(self._index)
        elif self._state is not None:
            self._state.next = self._stage(0)
            self._next = None
        else:
            self._next = None
        self._current = cache
        return cache

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        keep = False
        try:
            self._release()
            if exc_type is not None:
                self._synchronize()
            keep = exc_type is None and self._index == len(self._caches)
        finally:
            self._manager._finish(self._state, keep=keep)
            self._next = None
            self._state = None
        return False

    def _stage(self, index: int) -> tuple[Cache, torch.cuda.Event]:
        assert self._transfer_stream is not None
        with torch.cuda.stream(self._transfer_stream):
            if self._state is None:
                cache = self._caches[index].to(
                    self._device,
                    non_blocking=True,
                )
            else:
                slots = self._state.slots
                assert slots is not None
                slot = self._state.next_slot
                cache = slots[slot].copy_from(
                    self._caches[index],
                    non_blocking=True,
                )
                self._state.next_slot = 1 - slot
            ready = self._transfer_stream.record_event()
        return cache, ready

    def _release(self) -> None:
        if self._current is None:
            return
        assert self._compute_stream is not None
        assert self._transfer_stream is not None
        consumed = self._compute_stream.record_event()
        self._transfer_stream.wait_event(consumed)
        self._current = None

    def _synchronize(self) -> None:
        if self._compute_stream is not None:
            self._compute_stream.synchronize()
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()


class CacheManager:
    r"""Manage estimator cache placement and sequential device loading.

    CUDA ensembles use pinned host caches and two persistent device staging
    buffers. The first cache is prefetched for the next traversal while the
    final cache is consumed. Call :meth:`clear` to release the buffers.
    """

    def __init__(self) -> None:
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._state: _CacheState | None = None
        self._in_use = False

    def store(
        self,
        cache: Cache,
        *,
        num_estimators: int,
        device: torch.device,
    ) -> Cache:
        """Place an estimator cache for repeated prediction.

        Args:
            cache: Estimator cache produced during fitting.
            num_estimators: Number of fitted estimators.
            device: Device where the cache was produced.

        Returns:
            Cache placed on its prediction storage device.
        """
        if num_estimators == 1 or device.type != "cuda":
            return cache
        return cache.cpu().pin_memory()

    def load(
        self,
        caches: Sequence[Cache],
        device: torch.device,
    ) -> _CacheLoader:
        """Load estimator caches for sequential use on ``device``.

        The returned iterator must be used as a context manager. Each yielded
        cache remains valid until the iterator advances or its context exits.
        Source caches must remain unchanged between matching traversals.

        Args:
            caches: Estimator caches in prediction order.
            device: Device where each cache will be used.

        Returns:
            Context-managed iterator of caches placed on ``device``.
        """
        return _CacheLoader(self, caches, device)

    def clear(self) -> None:
        """Release staged device caches while retaining transfer streams."""
        self._discard_state()

    def _acquire(
        self,
        caches: Sequence[Cache],
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> _CacheState:
        if self._in_use:
            raise RuntimeError("CacheManager already has an active loader")
        if self._state is None or not self._state.matches(caches, device):
            self._discard_state()
            self._state = _CacheState(caches, device, stream)
        self._in_use = True
        return self._state

    def _finish(
        self,
        state: _CacheState | None,
        *,
        keep: bool,
    ) -> None:
        if state is not None:
            try:
                if not keep and self._state is state:
                    self._discard_state()
            finally:
                self._in_use = False

    def _invalidate_for(self, device: torch.device) -> None:
        if self._state is not None and self._state.device != device:
            self._discard_state()

    def _discard_state(self) -> None:
        state, self._state = self._state, None
        if state is None:
            return
        stream = self._streams.get(state.device)
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            state.dispose()

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self._streams[device] = stream
        return stream

    def __getstate__(self) -> dict[str, object]:
        for stream in self._streams.values():
            stream.synchronize()
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._streams = {}
        self._state = None
        self._in_use = False
