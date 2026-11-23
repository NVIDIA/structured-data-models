# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from enum import StrEnum
from typing import NamedTuple, Self, cast

import torch
from torch import Tensor

from sdm._memory import set_pinned_max_round_threshold
from sdm.tensor.mixin import DeviceMixin


class _QuantizedTensor(NamedTuple):
    qdata: Tensor
    scale: Tensor
    scheme: "QuantizationScheme"


class QuantizedTensor(_QuantizedTensor, DeviceMixin):
    r"""A tensor stored as quantized values and scales.

    Args:
        qdata: The quantized values.
        scale: The scales of ``qdata``.
        scheme: The quantization scheme of ``qdata`` and ``scale``.
    """

    def __init__(
        self,
        qdata: Tensor,
        scale: Tensor,
        scheme: "QuantizationScheme",
    ) -> None:
        if not scheme.is_valid(self):
            raise ValueError(
                f"Quantized values and scales do not match the quantization "
                f"scheme '{type(scheme).__name__}'"
            )

    @property
    def shape(self) -> torch.Size:
        r"""The shape of the tensor."""
        return self.qdata.shape

    def dequantize(self, dtype: torch.dtype) -> Tensor:
        r"""Returns the tensor in ``dtype``.

        Args:
            dtype: The target dtype.
        """
        return self.scheme.dequantize(self, dtype)

    def _tensors(self) -> Iterator[Tensor]:
        yield self.qdata
        yield self.scale

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.__class__(
            qdata=fn(self.qdata),
            scale=fn(self.scale),
            scheme=self.scheme,
        )


class _KVCacheEntry(NamedTuple):
    key: Tensor | QuantizedTensor
    value: Tensor | QuantizedTensor


class KVCacheEntry(_KVCacheEntry, DeviceMixin):
    r"""Keys and values of an attention layer.

    :class:`~sdm.nn.Attention` returns them for ``return_key_value=True`` and
    attends to them again when they are passed as ``key_value``.

    Args:
        key: Keys with shape ``[..., KV, H, C]``, possibly as a
            :class:`~sdm.nn.QuantizedTensor`.
        value: Values with shape ``[..., KV, H, C]``, possibly as a
            :class:`~sdm.nn.QuantizedTensor`.
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


class QuantizationScheme(abc.ABC):
    r"""Base class for storing tensors as quantized values and scales."""

    @abc.abstractmethod
    def quantize(self, tensor: Tensor) -> Tensor | QuantizedTensor:
        r"""Returns ``tensor`` stored in this scheme.

        Args:
            tensor: The tensor to quantize.
        """

    def dequantize(
        self,
        tensor: QuantizedTensor,
        dtype: torch.dtype,
    ) -> Tensor:
        r"""Returns ``tensor`` in ``dtype``.

        By default, it multiplies the quantized values by their scales.

        Args:
            tensor: A tensor stored in this scheme.
            dtype: The target dtype.
        """
        return tensor.qdata.to(dtype).mul_(tensor.scale)

    @abc.abstractmethod
    def is_valid(self, tensor: QuantizedTensor) -> bool:
        r"""Returns whether ``tensor`` holds values and scales of this scheme.

        Args:
            tensor: A quantized tensor.
        """


class Unquantized(QuantizationScheme):
    r"""Leaves tensors unquantized."""

    def quantize(self, tensor: Tensor) -> Tensor:  # noqa: D102
        return tensor

    def is_valid(self, tensor: QuantizedTensor) -> bool:  # noqa: D102
        return False


class Float8PerHead(QuantizationScheme):
    r"""Quantizes tensors to FP8 (E4M3) with one FP32 scale per head.

    Tensors have shape ``[..., S, H, C]`` and scales ``[..., 1, H, 1]``.
    """

    def quantize(self, tensor: Tensor) -> QuantizedTensor:  # noqa: D102
        amax = tensor.abs().amax(dim=(-3, -1), keepdim=True)  # [..., 1, H, 1]
        # Divide in the input dtype to avoid a full-size float32 copy:
        scale = (amax / 448).clamp_min(torch.finfo(tensor.dtype).tiny)
        qdata = (tensor / scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
        return QuantizedTensor(qdata=qdata, scale=scale.float(), scheme=self)

    def is_valid(self, tensor: QuantizedTensor) -> bool:  # noqa: D102
        qdata, scale = tensor.qdata, tensor.scale
        return (
            qdata.dtype == torch.float8_e4m3fn
            and scale.dtype == torch.float32
            and scale.shape == (*qdata.shape[:-3], 1, qdata.size(-2), 1)
        )


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
            self._kv_cache_dtype: QuantizationScheme = Unquantized()
        elif kv_cache_dtype == "fp8_per_head":
            self._kv_cache_dtype = Float8PerHead()
        else:
            raise ValueError(
                f"Unsupported key/value cache dtype '{kv_cache_dtype}'"
            )
        self._items: dict[Hashable, object] = dict(*args, **kwargs)

    @property
    def is_recording(self) -> bool:
        r"""Whether the cache is in recording mode."""
        return self._mode == Cache.Mode.record

    @property
    def is_replaying(self) -> bool:
        r"""Whether the cache is in replaying mode."""
        return self._mode == Cache.Mode.replay

    @property
    def kv_cache_dtype(self) -> QuantizationScheme:
        r"""The quantization scheme of cached keys and values."""
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
        self.__dict__.update(state)
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
        return out
