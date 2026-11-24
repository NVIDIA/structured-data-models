# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable, Iterator
from typing import NamedTuple, Self, cast

import torch
from torch import Tensor

from sdm.tensor.mixin import DeviceMixin


class _KeyValue(NamedTuple):
    key: Tensor
    value: Tensor
    key_scale: Tensor | None = None
    value_scale: Tensor | None = None
    kv_cache_dtype: "KVCacheDType | None" = None


class KeyValue(_KeyValue, DeviceMixin):
    r"""Key and value projections for attention, optionally quantized.

    Args:
        key: Keys with shape ``[..., KV, H, C]``.
        value: Values with shape ``[..., KV, H, C]``.
        key_scale: The scales of quantized ``key``.
        value_scale: The scales of quantized ``value``.
        kv_cache_dtype: The storage dtype of quantized ``key`` and ``value``.
    """

    def __new__(  # noqa: D102
        cls,
        key: Tensor,
        value: Tensor,
        key_scale: Tensor | None = None,
        value_scale: Tensor | None = None,
        kv_cache_dtype: "KVCacheDType | None" = None,
    ) -> Self:
        self = super().__new__(
            cls,
            key=key,
            value=value,
            key_scale=key_scale,
            value_scale=value_scale,
            kv_cache_dtype=kv_cache_dtype,
        )
        kv_cache_dtype = self.kv_cache_dtype or Unquantized()
        if not kv_cache_dtype.is_quantized(self):
            raise ValueError(
                f"K/V do not match the key/value cache dtype "
                f"'{type(kv_cache_dtype).__name__}'"
            )
        return self

    def dequantize(self, dtype: torch.dtype) -> "KeyValue":
        r"""Returns the keys and values, dequantized to ``dtype`` if quantized.

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
    r"""Base class for the storage dtype of key/value projections."""

    @abc.abstractmethod
    def is_quantized(self, key_value: KeyValue) -> bool:
        r"""Returns whether ``key_value`` is in this storage dtype.

        Args:
            key_value: Key/value projections.
        """

    @abc.abstractmethod
    def quantize(self, key_value: KeyValue) -> KeyValue:
        r"""Returns ``key_value`` in this storage dtype.

        Args:
            key_value: Key/value projections as computed by attention.
        """

    @abc.abstractmethod
    def dequantize(
        self,
        key_value: KeyValue,
        dtype: torch.dtype,
    ) -> KeyValue:
        r"""Returns ``key_value`` with keys and values in ``dtype``.

        Args:
            key_value: Key/value projections in this storage dtype.
            dtype: The target dtype.
        """


class Unquantized(KVCacheDType):
    r"""Leaves key/value projections as computed."""

    def is_quantized(self, key_value: KeyValue) -> bool:  # noqa: D102
        return key_value.key_scale is None and key_value.value_scale is None

    def quantize(self, key_value: KeyValue) -> KeyValue:  # noqa: D102
        return key_value

    def dequantize(  # noqa: D102
        self,
        key_value: KeyValue,
        dtype: torch.dtype,
    ) -> KeyValue:
        return key_value


class Float8PerHead(KVCacheDType):
    r"""Quantizes key/value to FP8 (E4M3) with one FP32 scale per head."""

    def is_quantized(self, key_value: KeyValue) -> bool:  # noqa: D102
        return all(
            data.dtype == torch.float8_e4m3fn
            and scale is not None
            and scale.dtype == torch.float32
            and scale.shape == (*data.shape[:-3], 1, data.size(-2), 1)
            for data, scale in (
                (key_value.key, key_value.key_scale),
                (key_value.value, key_value.value_scale),
            )
        )

    def quantize(self, key_value: KeyValue) -> KeyValue:  # noqa: D102
        if self.is_quantized(key_value):
            return key_value
        key, key_scale = self._quantize(key_value.key)
        value, value_scale = self._quantize(key_value.value)
        return KeyValue(
            key=key,
            value=value,
            key_scale=key_scale,
            value_scale=value_scale,
            kv_cache_dtype=self,
        )

    def dequantize(  # noqa: D102
        self,
        key_value: KeyValue,
        dtype: torch.dtype,
    ) -> KeyValue:
        return KeyValue(
            key=key_value.key.to(dtype).mul_(
                cast(Tensor, key_value.key_scale)
            ),
            value=key_value.value.to(dtype).mul_(
                cast(Tensor, key_value.value_scale)
            ),
        )

    @staticmethod
    def _quantize(tensor: Tensor) -> tuple[Tensor, Tensor]:
        amax = tensor.abs().amax(dim=(-3, -1), keepdim=True)  # [..., 1, H, 1]
        # Divide in the input dtype to avoid a full-size float32 copy:
        scale = (amax / 448).clamp_min(torch.finfo(tensor.dtype).tiny)
        data = (tensor / scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
        return data, scale.float()
