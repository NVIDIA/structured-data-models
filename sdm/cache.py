from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
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

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(key=fn(self.key), value=fn(self.value))


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

        out = self.__class__(
            {key: _apply(value) for key, value in self.items()}
        )
        out._mode = self._mode
        return out


class _CacheLoader(Iterator[Cache]):
    def __init__(
        self,
        caches: Sequence[Cache],
        device: torch.device,
        compute_stream: torch.cuda.Stream | None,
        transfer_stream: torch.cuda.Stream | None,
    ) -> None:
        self._caches = caches
        self._device = device
        self._compute_stream = compute_stream
        self._transfer_stream = transfer_stream
        self._index = 0

    def __next__(self) -> Cache:
        if self._index == len(self._caches):
            raise StopIteration
        cache = self._caches[self._index]
        self._index += 1
        if self._transfer_stream is None:
            return cache
        with torch.cuda.stream(self._transfer_stream):
            return cache.to(self._device, non_blocking=True)

    def wait(self) -> None:
        if self._compute_stream is not None:
            assert self._transfer_stream is not None
            self._compute_stream.wait_stream(self._transfer_stream)

    def record(self, cache: Cache) -> None:
        if self._compute_stream is not None:
            for tensor in cache._tensors():
                tensor.record_stream(self._compute_stream)


class CacheManager:  # noqa: D101
    def __init__(self) -> None:
        self._streams: dict[torch.device, torch.cuda.Stream] = {}

    def store(  # noqa: D102
        self,
        cache: Cache,
        *,
        num_estimators: int,
        device: torch.device,
    ) -> Cache:
        if num_estimators == 1 or device.type != "cuda":
            return cache
        return cache.cpu().pin_memory()

    @contextmanager
    def load(  # noqa: D102
        self,
        caches: Sequence[Cache],
        device: torch.device,
    ) -> Iterator[_CacheLoader]:
        compute_stream: torch.cuda.Stream | None = None
        transfer_stream: torch.cuda.Stream | None = None
        try:
            if device.type == "cuda":
                compute_stream = torch.cuda.current_stream(device)
                transfer_stream = self._stream(device)
            yield _CacheLoader(
                caches=caches,
                device=device,
                compute_stream=compute_stream,
                transfer_stream=transfer_stream,
            )
        except BaseException:
            if transfer_stream is not None:
                transfer_stream.synchronize()
            raise

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self._streams[device] = stream
        return stream

    def _synchronize(self) -> None:
        for stream in self._streams.values():
            stream.synchronize()

    def __getstate__(self) -> dict[str, object]:
        self._synchronize()
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._streams = {}
