# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import (
    Callable,
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

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(key=fn(self.key), value=fn(self.value))


def _quantize_int8(tensor: Tensor) -> tuple[Tensor, Tensor]:
    # Symmetric absmax quantization, with one scale per token and head. The
    # reduction reads `tensor` in its own dtype and only the small scale is
    # promoted, so no full-size floating-point copy is materialized.
    if tensor.numel() == 0:  # The reduction is undefined over an empty dim.
        return tensor.to(torch.int8), tensor.new_ones(
            (*tensor.size()[:-1], 1), dtype=torch.float32
        )
    absmax = torch.linalg.vector_norm(
        tensor,
        ord=torch.inf,
        dim=-1,
        keepdim=True,
    ).float()  # [..., KV, H, 1]
    scale = torch.where(absmax == 0, 1, absmax / 127)  # [..., KV, H, 1]
    quantized = (tensor / scale).round().clamp(-127, 127)
    return quantized.to(torch.int8), scale


def _dequantize_int8(
    tensor: Tensor,
    scale: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    # `tensor` is INT8 and `scale` FP32, so the product already promotes to
    # FP32 without an explicit cast of the full-size payload.
    return (tensor * scale).to(dtype)


class _Int8KVCacheEntry(NamedTuple):
    key: Tensor
    value: Tensor
    key_scale: Tensor
    value_scale: Tensor
    key_dtype: torch.dtype
    value_dtype: torch.dtype


class Int8KVCacheEntry(_Int8KVCacheEntry, DeviceMixin):
    r"""INT8 cached key/value projections for a single transformer block.

    Scales are computed independently for every batch element, token and
    attention head, and shared across the channels of a head.

    Args:
        key: INT8 key payload with shape ``[..., KV, H, C]``.
        value: INT8 value payload with shape ``[..., KV, H, C]``.
        key_scale: FP32 key scale with shape ``[..., KV, H, 1]``.
        value_scale: FP32 value scale with shape ``[..., KV, H, 1]``.
        key_dtype: Dtype restored when dequantizing the key.
        value_dtype: Dtype restored when dequantizing the value.
    """

    @classmethod
    def from_entry(cls, entry: KVCacheEntry) -> Self:
        r"""Quantize floating-point key/value projections.

        Args:
            entry: Key/value projections with shape ``[..., KV, H, C]``.

        Returns:
            Quantized key/value projections.
        """
        key, key_scale = _quantize_int8(entry.key)
        value, value_scale = _quantize_int8(entry.value)
        return cls(
            key=key,
            value=value,
            key_scale=key_scale,
            value_scale=value_scale,
            key_dtype=entry.key.dtype,
            value_dtype=entry.value.dtype,
        )

    def dequantize(self) -> KVCacheEntry:
        r"""Restore floating-point key/value projections.

        Returns:
            Key/value projections with their original dtypes and shapes.
        """
        return KVCacheEntry(
            key=_dequantize_int8(self.key, self.key_scale, self.key_dtype),
            value=_dequantize_int8(
                self.value,
                self.value_scale,
                self.value_dtype,
            ),
        )

    def _tensors(self) -> Iterator[Tensor]:
        yield self.key
        yield self.value
        yield self.key_scale
        yield self.value_scale

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            key=fn(self.key),
            value=fn(self.value),
            key_scale=fn(self.key_scale),
            value_scale=fn(self.value_scale),
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
        )


class Cache(MutableMapping[Hashable, object], DeviceMixin):
    r"""A mutable mapping of model cache values.

    Args:
        args: Initial cache data as a mapping or iterable of key/value pairs.
        kv_cache_dtype: Storage dtype for recorded key/value projections. Pass
            :external+torch:ref:`torch.int8 <dtype-doc>` to store approximate
            one-byte payloads with FP32 scales, or ``None`` to preserve the
            projected dtype.
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
        kv_cache_dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> None:
        if kv_cache_dtype not in (None, torch.int8):
            raise ValueError(
                f"Unsupported key/value cache dtype '{kv_cache_dtype}'"
            )
        self._mode = Cache.Mode.record
        self._kv_cache_dtype = kv_cache_dtype
        self._items: dict[Hashable, object] = {}
        for key, value in dict(*args, **kwargs).items():
            self[key] = value  # Quantizes key/value entries, if requested.

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
        if self._kv_cache_dtype == torch.int8 and isinstance(
            value, KVCacheEntry
        ):
            value = Int8KVCacheEntry.from_entry(value)
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
        r"""Return an empty recording cache with the same configuration."""
        return self.__class__(kv_cache_dtype=self._kv_cache_dtype)
