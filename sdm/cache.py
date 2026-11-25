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
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, Self, cast, overload

import torch
from torch import Tensor

from sdm._memory import set_pinned_max_round_threshold
from sdm.tensor.mixin import DeviceMixin
from sdm.tensor.var_len import _mark_dynamic


class _KVCacheEntry(NamedTuple):
    key: "Tensor | QuantizedTensor"
    value: "Tensor | QuantizedTensor"


class KVCacheEntry(_KVCacheEntry, DeviceMixin):
    r"""Cached key/value projections for a single transformer block.

    Args:
        key: Key projections with shape ``[..., KV, H, C]``.
        value: Value projections with shape ``[..., KV, H, C]``.
    """

    def _tensors(self) -> Iterator[Tensor]:
        for tensor in (self.key, self.value):
            if isinstance(tensor, QuantizedTensor):
                yield from tensor._tensors()
            else:
                yield tensor

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        def apply(
            tensor: Tensor | QuantizedTensor,
        ) -> Tensor | QuantizedTensor:
            if isinstance(tensor, QuantizedTensor):
                return tensor._apply_tensor(fn)
            return fn(tensor)

        return self.__class__(key=apply(self.key), value=apply(self.value))


@dataclass(frozen=True, eq=False)
class QuantizedTensor(DeviceMixin):
    r"""A tensor stored as FP8 (E4M3) values and FP32 scales.

    Args:
        qdata: The quantized values.
        scale: The scales of ``qdata``, broadcastable to it.
    """

    qdata: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if (
            self.qdata.dtype != torch.float8_e4m3fn
            or self.scale.dtype != torch.float32
        ):
            raise ValueError(
                f"Expected FP8 (E4M3) values and FP32 scales (got "
                f"'{self.qdata.dtype}' and '{self.scale.dtype}')"
            )

    @classmethod
    def quantize(cls, tensor: Tensor, dim: int | tuple[int, ...]) -> Self:
        r"""Quantizes ``tensor`` scaled by its absolute maximum over ``dim``.

        Args:
            tensor: The tensor to quantize.
            dim: The dimensions that share a scale.
        """
        amax = tensor.abs().amax(dim=dim, keepdim=True)
        # Divide in the input dtype to avoid a full-size float32 copy:
        scale = (amax / 448).clamp_min(torch.finfo(tensor.dtype).tiny)
        qdata = (tensor / scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
        return cls(qdata=qdata, scale=scale.float())

    @overload
    def size(self) -> torch.Size: ...

    @overload
    def size(self, dim: int) -> int: ...

    def size(self, dim: int | None = None) -> torch.Size | int:
        r"""Returns the size of the quantized values.

        Args:
            dim: The dimension to return the size of, or all if ``None``.
        """
        return self.qdata.size(dim)

    def dequantize(self, dtype: torch.dtype) -> Tensor:
        r"""Returns the tensor in ``dtype``.

        Args:
            dtype: The target dtype.
        """
        return self.qdata.to(dtype).mul_(self.scale)

    def _tensors(self) -> Iterator[Tensor]:
        yield self.qdata
        yield self.scale

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(qdata=fn(self.qdata), scale=fn(self.scale))


class KVCacheDType(abc.ABC):
    r"""Base class for the storage dtype of cached key/value projections."""

    @abc.abstractmethod
    def quantize(
        self,
        tensor: Tensor | QuantizedTensor,
    ) -> Tensor | QuantizedTensor:
        r"""Returns ``tensor`` in this storage dtype, or as is if quantized.

        Args:
            tensor: Key or value projections as computed by attention.
        """


class Unquantized(KVCacheDType):
    r"""Leaves key/value projections as computed."""

    def quantize(  # noqa: D102
        self,
        tensor: Tensor | QuantizedTensor,
    ) -> Tensor | QuantizedTensor:
        return tensor


class Float8PerHead(KVCacheDType):
    r"""Quantizes key/value to FP8 (E4M3) with one FP32 scale per head."""

    def quantize(  # noqa: D102
        self,
        tensor: Tensor | QuantizedTensor,
    ) -> QuantizedTensor:
        if isinstance(tensor, QuantizedTensor):
            return tensor
        return QuantizedTensor.quantize(tensor, dim=(-3, -1))  # [..., 1, H, 1]


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
        # The dimensions of entries that follow the fitted data; see `_record`.
        self._dims: dict[Hashable, Mapping[int | None, int]] = {}
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

    def _record(
        self,
        key: Hashable,
        value: KVCacheEntry,
        dims: Mapping[int | None, int],
    ) -> None:
        r"""Store ``value`` and declare its dimensions that follow the table.

        Compiled predictions keep them dynamic along with those of the query.

        Args:
            key: The cache key.
            value: The keys and values.
            dims: The dimensions of ``value`` by the dimension of the table,
                ``[..., R, D]``, that they follow, *e.g.*, ``{-2: -3}`` for
                keys and values of shape ``[..., R, H, C]``.
        """
        self[key] = value
        self._dims[key] = dims

    def _dynamic_dims(
        self,
    ) -> Iterator[tuple[Tensor, Mapping[int | None, int]]]:
        # Every fitted tensor with its declared dimensions; see `_record`.
        for key, value in self.items():
            dims = self._dims.get(key)
            if dims is not None:
                # Keys and values of `_record`, not the scales of quantized
                # ones, which are per head.
                entry = cast(KVCacheEntry, value)
                for tensor in (entry.key, entry.value):
                    if isinstance(tensor, QuantizedTensor):
                        tensor = tensor.qdata
                    yield tensor, dims
            else:
                # A nested cache or the recipe execution lists its own.
                declared = getattr(value, "_dynamic_dims", None)
                if declared is not None:
                    yield from declared()

    def _mark_dynamic(self, dims: Collection[int]) -> None:
        r"""Mark the declared dimensions that follow ``dims`` as dynamic.

        Marks for :func:`torch.compile` stay on the tensors, so a cache is
        marked once.

        Args:
            dims: The dimensions of a table, counted from its last one,
                *e.g.*, ``{-2}`` for the rows.
        """
        if self._dynamic:
            return
        for tensor, follows in self._dynamic_dims():
            for table_dim, dim in follows.items():
                # `None` stands for a dimension that follows the fitted data
                # itself, such as a list of categories.
                if table_dim is None or table_dim in dims:
                    _mark_dynamic(tensor, dim)
        self._dynamic = True

    def __setitem__(self, key: Hashable, value: object) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__setitem__' requires the cache to be in 'record' mode"
            )
        self._items[key] = value
        self._dims.pop(key, None)

    def __delitem__(self, key: Hashable) -> None:
        if not self.is_recording:
            raise RuntimeError(
                "'__delitem__' requires the cache to be in 'record' mode"
            )
        del self._items[key]
        self._dims.pop(key, None)

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
        out._dims = dict(self._dims)
        return out
