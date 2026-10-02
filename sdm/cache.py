# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
            elif isinstance(value, (list, tuple)):
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


class _CachePrefetcher:
    r"""Iterate over caches while transferring the next cache to CUDA."""

    def __init__(
        self,
        caches: Sequence[Cache],
        device: torch.device,
        transfer_stream: torch.cuda.Stream | None,
    ) -> None:
        if len(caches) == 0:
            raise ValueError("Expected at least one cache to prefetch")
        self._caches = caches
        self._device = device
        self._transfer_stream = transfer_stream
        self._compute_stream: torch.cuda.Stream | None = None
        self._next_cache: Cache | None = None

    def __enter__(self) -> Self:
        if self._device.type == "cuda":
            assert self._transfer_stream is not None
            self._compute_stream = torch.cuda.current_stream(self._device)
            with torch.cuda.stream(self._transfer_stream):
                self._next_cache = self._caches[0].to(
                    device=self._device,
                    non_blocking=True,
                )
        else:
            self._next_cache = self._caches[0]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None and self._transfer_stream is not None:
            self._transfer_stream.synchronize()

    def __iter__(self) -> Iterator[Cache]:
        for i in range(len(self._caches)):
            cache, self._next_cache = self._next_cache, None
            assert cache is not None

            if self._compute_stream is not None:
                assert self._transfer_stream is not None
                self._compute_stream.wait_stream(self._transfer_stream)
                # Record before yielding so an exception in the consumer
                # cannot release tensors still used by queued compute.
                for tensor in cache._tensors():
                    tensor.record_stream(self._compute_stream)

            if i + 1 < len(self._caches):
                self._next_cache = self._caches[i + 1]
            if (
                self._transfer_stream is not None
                and self._next_cache is not None
            ):
                with torch.cuda.stream(self._transfer_stream):
                    self._next_cache = self._next_cache.to(
                        device=self._device,
                        non_blocking=True,
                    )

            yield cache
