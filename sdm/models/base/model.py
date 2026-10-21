# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Iterator, Sequence
from typing import ClassVar, Generic, Self, TypeVar

import torch
from torch import Tensor

from sdm import Recipe, Stype
from sdm.cache import Cache
from sdm.tensor.mixin import DeviceMixin

T = TypeVar("T", bound=Tensor | DeviceMixin)


class Model(torch.nn.Module, abc.ABC):
    r"""Base model for foundation models on structured data.

    :class:`Model` extends :class:`torch.nn.Module` by unified pre-processing
    and post-processing routines, caching, and ensembling.
    """

    #: Semantic types supported for input columns in this model.
    supported_feature_stypes: ClassVar[frozenset[Stype]]

    def __init__(self) -> None:
        super().__init__()

        self._cache: Cache | None = None
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    def clear(self) -> None:
        r"""Clear any cached state."""
        self._cache = None

    # Helpers #################################################################

    def _prefetch(
        self,
        items: Sequence[T],
        device: torch.device,
    ) -> "_Prefetcher[T]":

        stream: torch.cuda.Stream | None = None
        if device.type == "cuda":
            if device not in self._transfer_streams:
                self._transfer_streams[device] = torch.cuda.Stream(device)
            stream = self._transfer_streams[device]

        return _Prefetcher(items, device, stream)

    # Python Builtins #########################################################

    def __getstate__(self) -> dict[str, object]:
        for stream in self._transfer_streams.values():
            stream.synchronize()
        state = super().__getstate__()
        state.pop("_transfer_streams", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._transfer_streams = {}

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"


class _Prefetcher(Generic[T]):
    def __init__(
        self,
        items: Sequence[T],
        device: torch.device,
        stream: torch.cuda.Stream | None,
    ) -> None:
        if len(items) == 0:
            raise ValueError("Expected at least one item to prefetch")

        self._items = items
        self._device = device
        self._transfer_stream = stream

        self._compute_stream: torch.cuda.Stream | None = None
        self._next: T | None = None

    def __enter__(self) -> Self:
        if self._transfer_stream is not None:
            self._compute_stream = torch.cuda.current_stream(self._device)
            try:
                with torch.cuda.stream(self._transfer_stream):
                    self._next = self._items[0].to(  # type: ignore
                        device=self._device,
                        non_blocking=True,
                    )
            except BaseException:
                self._transfer_stream.synchronize()
                raise
        else:
            self._next = self._items[0].to(self._device)  # type: ignore

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        *args: object,
    ) -> None:
        if exc_type is not None and self._transfer_stream is not None:
            self._transfer_stream.synchronize()

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self._items)):
            item, self._next = self._next, None
            assert item is not None

            if self._transfer_stream is not None:
                assert self._compute_stream is not None
                event = self._transfer_stream.record_event()
                self._compute_stream.wait_event(event)
                if isinstance(item, DeviceMixin):
                    for tensor in item._tensors():
                        tensor.record_stream(self._compute_stream)
                else:
                    item.record_stream(self._compute_stream)

            if i + 1 < len(self._items):
                self._next = self._items[i + 1]

            if self._next is not None:
                if self._transfer_stream is not None:
                    with torch.cuda.stream(self._transfer_stream):
                        self._next = self._next.to(  # type: ignore
                            device=self._device,
                            non_blocking=True,
                        )
                else:
                    self._next = self._next.to(self._device)  # type: ignore

            yield item
