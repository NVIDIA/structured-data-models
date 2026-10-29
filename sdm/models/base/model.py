# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Generic, Protocol, Self, TypeVar

import torch
from torch import Tensor

from sdm import Recipe, Stype, TableTensor
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.tensor.mixin import DeviceMixin


class _Execution(Protocol):
    def _check_model_call(self, operation: str) -> None: ...
    def _prepared_forward(
        self, forward: Callable[..., Any], **kwargs: Any
    ) -> Any: ...
    def _run_core(
        self, task: Any, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> Tensor: ...


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
        self._execution: _Execution | None = None
        self._transfer_streams: dict[torch.device, torch.cuda.Stream] = {}

    @classmethod
    @abc.abstractmethod
    def default_recipe(cls) -> Recipe:
        r"""Return the default processing recipe for this model."""

    def clear(self) -> None:
        r"""Clear any cached state."""
        self._check_execution("clear")
        self._cache = None

    def _check_execution(self, operation: str) -> None:
        if self._execution is not None:
            self._execution._check_model_call(operation)

    def train(self, mode: bool = True) -> Self:
        """Set training mode when no executor owns this model."""
        self._check_execution("train")
        return super().train(mode)

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> Self:
        self._check_execution("_apply")
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Load weights when no executor owns this model."""
        self._check_execution("load_state_dict")
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

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
        self._check_execution("serialize")
        for stream in self._transfer_streams.values():
            stream.synchronize()
        state = super().__getstate__()
        state.pop("_transfer_streams", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._transfer_streams = {}
        self._execution = None

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"

    # Helpers #################################################################

    def _validate_feature_stype(self, x: TableTensor) -> None:
        invalid = x.active_stypes - self.supported_feature_stypes - {Stype.id}
        if len(invalid) > 0:
            stypes = ", ".join(f"{str(stype)!r}" for stype in invalid)
            warn_once(
                key="model-unsupported-feature-stypes",
                message=(
                    f"{self.__class__.__name__!r} received unsupported "
                    f"feature stypes {stypes}. Columns with unsupported "
                    f"feature stypes will not be consumed by the model."
                ),
            )


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
