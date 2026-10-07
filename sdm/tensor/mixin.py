# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable, Iterator
from typing import Any, Self, cast

import torch
from torch import Tensor

from sdm.tensor.var_len import VarLenTensor


class DeviceMixin(abc.ABC):
    r"""Add :class:`torch.Tensor`-like device capabilities to an object."""

    def to(
        self,
        device: torch.device | str | None,
        *,
        non_blocking: bool = False,
    ) -> Self:
        r"""Perform device conversion.

        Args:
            device: The device.
            non_blocking: If ``True`` and the source is in pinned memory, the
                copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        return self._apply_tensor(
            lambda x: x.to(device, non_blocking=non_blocking)
        )

    def cpu(self) -> Self:
        r"""Copy data in CPU memory."""
        return self._apply_tensor(lambda x: x.cpu())

    def cuda(
        self,
        device: torch.device | str | int | None = None,
        *,
        non_blocking: bool = False,
    ) -> Self:
        r"""Copy data in CUDA memory.

        Args:
            device: The device.
            non_blocking: If ``True`` and the source is in pinned memory, the
                copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        return self._apply_tensor(
            lambda x: x.cuda(device, non_blocking=non_blocking)
        )

    @property
    def device(self) -> torch.device:
        r"""The :class:`torch.device` where the data is."""
        devices = {tensor.device for tensor in self._tensors()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"{self.__class__.__name__!r}"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected tensors in {self.__class__.__name__!r} to be on "
                f"the same device (got {list(devices)})"
            )
        return next(iter(devices))

    @property
    def is_cpu(self) -> bool:
        r"""Whether the data is stored on the CPU."""
        devices = {tensor.device for tensor in self._tensors()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'is_cpu' of empty "
                f"{self.__class__.__name__!r}"
            )
        return all(device.type == "cpu" for device in devices)

    @property
    def is_cuda(self) -> bool:
        r"""Whether the data is stored on the GPU."""
        devices = {tensor.device for tensor in self._tensors()}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'is_cuda' of empty "
                f"{self.__class__.__name__!r}"
            )
        return all(device.type == "cuda" for device in devices)

    def pin_memory(self) -> Self:
        r"""Copy data to pinned memory, if it is not already pinned."""
        return self._apply_tensor(lambda x: x.pin_memory())

    def is_pinned(self) -> bool:
        r"""Returns ``True`` if data resides in pinned memory."""
        tensors = tuple(tensor for tensor in self._tensors())
        if len(tensors) == 0:
            raise RuntimeError(
                f"Could not determine 'in_pinned' of empty "
                f"{self.__class__.__name__!r}"
            )
        return all(tensor.is_pinned() for tensor in tensors)

    @abc.abstractmethod
    def _tensors(self) -> Iterator[Tensor]:
        pass

    @abc.abstractmethod
    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        pass


def _resolve_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.type == "mps" and device.index is None:
        return torch.device("mps", 0)
    return device


def _copy_wrapper_(
    destination: Tensor,
    source: Tensor,
    non_blocking: bool = False,
) -> Tensor:
    """Copy matching encoded storage while retaining tensor aliases."""
    leaves: dict[int, tuple[Tensor, Tensor]] = {}

    def collect(dst: Tensor, src: Tensor) -> None:
        if dst.shape != src.shape:
            raise ValueError("Copy requires matching tensor storage shapes")
        if hasattr(dst, "__tensor_flatten__"):
            if type(dst) is not type(src):
                raise TypeError(
                    "Copy requires matching tensor container types"
                )
            dst_names, dst_context = cast(Any, dst).__tensor_flatten__()
            src_names, src_context = cast(Any, src).__tensor_flatten__()
            if dst_names != src_names or dst_context != src_context:
                raise ValueError(
                    "Copy requires matching tensor container metadata"
                )
            if isinstance(dst, VarLenTensor):
                if (
                    dst.stride() != src.stride()
                    or dst.storage_offset() != src.storage_offset()
                ):
                    raise ValueError("Copy requires matching ragged layouts")
                # Layout only carries view metadata; copying the canonical
                # offset buffer also updates this alias.
                dst_names = [name for name in dst_names if name != "_layout"]
            for name in dst_names:
                collect(getattr(dst, name), getattr(src, name))
        else:
            if id(dst) in leaves and leaves[id(dst)][1] is not src:
                raise ValueError(
                    "Copy requires matching aliased tensor leaves"
                )
            leaves[id(dst)] = (dst, src)

    collect(destination, source)
    copies = [(dst, src) for dst, src in leaves.values() if dst is not src]
    for dst, _ in copies:
        for other_dst, _ in leaves.values():
            if dst is not other_dst and torch._C._is_alias_of(dst, other_dst):
                raise ValueError(
                    "Copy does not support distinct destination leaves "
                    "sharing storage"
                )
    for i, (dst, _) in enumerate(copies):
        for j, (_, src) in enumerate(copies):
            if i != j and torch._C._is_alias_of(dst, src):
                raise ValueError(
                    "Copy does not support cross-leaf storage overlap"
                )
    for dst, src in copies:
        dst.copy_(src, non_blocking=non_blocking)
    return destination
