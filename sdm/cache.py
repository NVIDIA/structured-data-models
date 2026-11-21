# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from enum import StrEnum
from typing import Any, NamedTuple, Self, cast

import torch
from torch import Tensor

from sdm._memory import set_pinned_max_round_threshold
from sdm.tensor.mixin import DeviceMixin
from sdm.tensor.var_len import _mark_dynamic


class _KVCacheEntry(NamedTuple):
    key: Tensor
    value: Tensor
    key_scale: Tensor | None = None
    value_scale: Tensor | None = None
    kv_cache_dtype: "KVCacheDType | None" = None


class KVCacheEntry(_KVCacheEntry, DeviceMixin):
    r"""Cached key/value projections for a single transformer block.

    Args:
        key: Key projections with shape ``[..., KV, H, C]``.
        value: Value projections with shape ``[..., KV, H, C]``.
        key_scale: Optional scales broadcastable to ``key``.
        value_scale: Optional scales broadcastable to ``value``.
        kv_cache_dtype: The storage dtype of K/V, if any.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:  # noqa: D102
        self = super().__new__(cls, *args, **kwargs)
        kv_cache_dtype = self.kv_cache_dtype or Unquantized()
        if not kv_cache_dtype.is_quantized(self):
            raise ValueError(
                f"K/V do not match the key/value cache dtype "
                f"'{type(kv_cache_dtype).__name__}'"
            )
        return self

    def dequantize(self, dtype: torch.dtype) -> "KVCacheEntry":
        r"""Returns the entry with quantized K/V dequantized to ``dtype``.

        Args:
            dtype: The target dtype.
        """
        if self.kv_cache_dtype is None:
            return self
        return self.kv_cache_dtype.dequantize(self, dtype)

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value
        if self.key_scale is not None:
            yield self.key_scale
        if self.value_scale is not None:
            yield self.value_scale

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            key=fn(self.key),
            value=fn(self.value),
            key_scale=None if self.key_scale is None else fn(self.key_scale),
            value_scale=None
            if self.value_scale is None
            else fn(self.value_scale),
            kv_cache_dtype=self.kv_cache_dtype,
        )


class KVCacheDType(abc.ABC):
    r"""Base class for the storage dtype of cached key/value projections."""

    @abc.abstractmethod
    def is_quantized(self, entry: KVCacheEntry) -> bool:
        r"""Returns whether K/V of ``entry`` are in this storage dtype.

        Args:
            entry: Key/value projections.
        """

    @abc.abstractmethod
    def quantize(self, entry: KVCacheEntry) -> KVCacheEntry:
        r"""Returns ``entry`` in this storage dtype.

        Args:
            entry: Key/value projections as computed by attention.
        """

    @abc.abstractmethod
    def dequantize(
        self,
        entry: KVCacheEntry,
        dtype: torch.dtype,
    ) -> KVCacheEntry:
        r"""Returns ``entry`` with key/value projections in ``dtype``.

        Args:
            entry: Key/value projections in this storage dtype.
            dtype: The target dtype.
        """


class Unquantized(KVCacheDType):
    r"""Leaves key/value projections as computed."""

    def is_quantized(self, entry: KVCacheEntry) -> bool:  # noqa: D102
        return entry.key_scale is None and entry.value_scale is None

    def quantize(self, entry: KVCacheEntry) -> KVCacheEntry:  # noqa: D102
        return entry

    def dequantize(  # noqa: D102
        self,
        entry: KVCacheEntry,
        dtype: torch.dtype,
    ) -> KVCacheEntry:
        return entry


class Float8PerHead(KVCacheDType):
    r"""Quantizes key/value to FP8 (E4M3) with one FP32 scale per head."""

    def is_quantized(self, entry: KVCacheEntry) -> bool:  # noqa: D102
        return all(
            data.dtype == torch.float8_e4m3fn
            and scale is not None
            and scale.dtype == torch.float32
            and scale.shape == (*data.shape[:-3], 1, data.size(-2), 1)
            for data, scale in (
                (entry.key, entry.key_scale),
                (entry.value, entry.value_scale),
            )
        )

    def quantize(self, entry: KVCacheEntry) -> KVCacheEntry:  # noqa: D102
        if self.is_quantized(entry):
            return entry
        key, key_scale = self._quantize(entry.key)
        value, value_scale = self._quantize(entry.value)
        return KVCacheEntry(
            key=key,
            value=value,
            key_scale=key_scale,
            value_scale=value_scale,
            kv_cache_dtype=self,
        )

    def dequantize(  # noqa: D102
        self,
        entry: KVCacheEntry,
        dtype: torch.dtype,
    ) -> KVCacheEntry:
        return KVCacheEntry(
            key=entry.key.to(dtype).mul_(cast(Tensor, entry.key_scale)),
            value=entry.value.to(dtype).mul_(cast(Tensor, entry.value_scale)),
        )

    @staticmethod
    def _quantize(tensor: Tensor) -> tuple[Tensor, Tensor]:
        amax = tensor.abs().amax(dim=(-3, -1), keepdim=True)  # [..., 1, H, 1]
        # Divide in the input dtype to avoid a full-size float32 copy:
        scale = (amax / 448).clamp_min(torch.finfo(tensor.dtype).tiny)
        data = (tensor / scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
        return data, scale.float()


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
        # TODO: Introduce registry to map str to quantization
        kv_cache_dtype = kwargs.pop("kv_cache_dtype", None)
        if kv_cache_dtype is None:
            self._kv_cache_dtype: KVCacheDType = Unquantized()
        elif kv_cache_dtype == "fp8_per_head":
            self._kv_cache_dtype = Float8PerHead()
        else:
            raise ValueError(
                f"Unsupported key/value cache dtype '{kv_cache_dtype}'"
            )
        self._items: dict[Hashable, object] = dict(*args, **kwargs)
        # The dimensions of entries that follow the fitted data; see `record`.
        self._dims: dict[Hashable, dict[str, int]] = {}
        self._dynamic = False

    @property
    def is_recording(self) -> bool:
        r"""Whether the cache is in recording mode."""
        return self._mode == Cache.Mode.record

    @property
    def is_replaying(self) -> bool:
        r"""Whether the cache is in replaying mode."""
        return self._mode == Cache.Mode.replay

    @property
    def kv_cache_dtype(self) -> KVCacheDType:
        r"""The storage dtype of cached key/value projections."""
        return self._kv_cache_dtype

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

    def record(
        self,
        key: Hashable,
        value: Tensor | KVCacheEntry,
        **dims: int,
    ) -> None:
        r"""Store ``value`` and declare the dimensions of its tensors.

        The dimensions follow the fitted data, and compiled predictions keep
        them dynamic.

        Args:
            key: The cache key.
            value: A tensor or a :class:`KVCacheEntry`.
            **dims: The dimensions by name, *e.g.*, ``rows=-3`` for keys and
                values of shape ``[..., R, H, C]`` with one entry per row.
        """
        self[key] = value
        self._dims[key] = dims

    def _dynamic_dims(self) -> Iterator[tuple[Tensor, Mapping[str, int]]]:
        # The recorded dimensions, and those that other entries declare, such
        # as nested caches and the recipe execution.
        for key, value in self.items():
            dims = self._dims.get(key)
            if dims is None:
                declared = getattr(value, "_dynamic_dims", None)
                if declared is not None:
                    yield from declared()
            elif isinstance(value, DeviceMixin):
                for tensor in value._tensors():
                    yield tensor, dims
            else:
                yield cast(Tensor, value), dims

    def _mark_dynamic(self, names: Collection[str]) -> None:
        r"""Mark the declared dimensions called ``names`` as dynamic.

        Marks for :func:`torch.compile` stay on the tensors, so a cache is
        marked once.

        Args:
            names: The names of the dimensions, *e.g.*, ``{"rows"}``.
        """
        if self._dynamic:
            return
        for tensor, dims in self._dynamic_dims():
            for name, dim in dims.items():
                if name in names:
                    _mark_dynamic(tensor, dim)
        self._dynamic = True

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

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Stable serialization metadata: positions in the tensor traversal.
        state["_pinned_tensor_indices"] = tuple(
            index
            for index, tensor in enumerate(self._tensors())
            if tensor.is_pinned()
        )
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        pinned = set(
            cast(tuple[int, ...], state.pop("_pinned_tensor_indices", ()))
        )
        # Caches saved before they recorded dimensions have none, and the
        # marks of a saved cache do not survive its tensors.
        self.__dict__.update({"_dims": {}} | state)
        self._dynamic = False
        if not pinned or not torch.cuda.is_available():
            return

        replacements: dict[int, Tensor] = {}
        for index, tensor in enumerate(self._tensors()):
            if (
                index in pinned
                and tensor.is_cpu
                and not tensor.is_pinned()
                and id(tensor) not in replacements
            ):
                if not replacements:
                    set_pinned_max_round_threshold(mb=1)
                replacements[id(tensor)] = tensor.pin_memory()

        if replacements:
            self._items = self._apply_tensor(
                lambda tensor: replacements.get(id(tensor), tensor)
            )._items

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
        out._kv_cache_dtype = self._kv_cache_dtype
        out._mode = self._mode
        out._dims = self._dims
        return out
