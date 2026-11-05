# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, Self, TypeVar, cast

import torch
from torch import Tensor

from sdm import EnsembleTensor

T = TypeVar("T", bound="Tensor | BufferList[Any]")


class _EnsembleBuffer(torch.nn.Module, EnsembleTensor[Tensor]):
    """Adapt ensemble tensor storage to PyTorch's module lifecycle."""

    def __init__(
        self,
        groups: Sequence[Tensor],
        locations: Sequence[tuple[int, int]],
        *,
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self._groups = tuple(groups)
        self._locations = tuple(locations)
        self._persistent = persistent
        for index, group in enumerate(groups):
            self.register_buffer(str(index), group, persistent=persistent)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> Self:
        super()._apply(fn, recurse=recurse)
        self._groups = tuple(
            cast(Tensor, group) for group in self._buffers.values()
        )
        return self

    def get_extra_state(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        return len(self._groups), self._locations

    def set_extra_state(self, state: object) -> None:
        _, self._locations = cast(
            tuple[int, tuple[tuple[int, int], ...]], state
        )

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        if self._persistent:
            super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = f"{prefix}_extra_state"
        if self._persistent and key in state_dict:
            num_groups, _ = state_dict[key]
            previous = self._buffers.copy()
            self._buffers.clear()
            for index in range(num_groups):
                name = str(index)
                saved = state_dict.get(f"{prefix}{name}")
                buffer = previous.get(name)
                if isinstance(saved, Tensor):
                    if buffer is None:
                        buffer = torch.empty_like(saved)
                    elif buffer.shape != saved.shape:
                        buffer = buffer.new_empty(saved.shape)
                elif buffer is None:
                    buffer = torch.empty(0)
                self.register_buffer(name, buffer)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not self._persistent and key in missing_keys:
            missing_keys.remove(key)
        self._groups = tuple(
            cast(Tensor, group) for group in self._buffers.values()
        )


class BufferList(torch.nn.Module, Generic[T]):
    """Ordered list of tensors registered as PyTorch module state.

    Args:
        buffers: Tensors and buffer lists to register in order.
    """

    def __init__(
        self,
        buffers: Iterable[T] = (),
    ) -> None:
        super().__init__()
        for index, item in enumerate(buffers):
            if isinstance(item, BufferList):
                self.add_module(str(index), item)
            else:
                self.register_buffer(str(index), item)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        layout = state_dict[f"{prefix}_extra_state"]
        materialized_keys = []
        for index, is_nested in enumerate(layout):
            if is_nested:
                continue
            name = str(index)
            key = f"{prefix}{index}"
            if name not in self._buffers:
                # Some tensor subclasses cannot use empty_like or copy_.
                self.register_buffer(name, state_dict.pop(key).clone())
                materialized_keys.append(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in materialized_keys:
            if key in missing_keys:
                missing_keys.remove(key)

    def __getitem__(self, index: int) -> T:
        name = str(index)
        if name in self._modules:
            return cast(T, self.get_submodule(name))
        return cast(T, self.get_buffer(name))

    def __iter__(self) -> Iterator[T]:
        return (self[index] for index in range(len(self)))

    def __len__(self) -> int:
        return len(self._buffers) + len(self._modules)

    def get_extra_state(self) -> tuple[bool, ...]:
        r""":meta private:"""  # noqa: D415
        # Layout of the buffer list in a tuple of booleans.
        # True at positions that hold a nested BufferList rather than a tensor.
        return tuple(str(index) in self._modules for index in range(len(self)))

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        for index, is_nested in enumerate(cast(tuple[bool, ...], state)):
            name = str(index)
            if is_nested and name not in self._modules:
                self.add_module(name, BufferList())
