# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import torch
from torch import Tensor

from sdm import EnsembleData, EnsembleTensor


class _EnsembleTensorBuffer(torch.nn.Module, EnsembleTensor[Tensor]):
    """Ensemble tensor registered as PyTorch module state."""

    def __getattribute__(self, name: str) -> Any:
        # State remains tensor-like even where nn.Module has methods such as
        # to(), cpu() and type(). Parent module transfers still use _apply().
        if (
            not name.startswith("_")
            and name in torch.nn.Module.__dict__
            and callable(getattr(Tensor, name, None))
        ):
            return functools.partial(
                EnsembleData.execute, getattr(Tensor, name), self
            )
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return EnsembleTensor.__getattr__(self, name)

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
            for name, group in self._buffers.items():
                if isinstance(group, Tensor) and type(group) is not Tensor:
                    destination[f"{prefix}{name}"] = (
                        group if keep_vars else group.clone()
                    )

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
        materialized = []
        if self._persistent and key in state_dict:
            num_groups, _ = state_dict[key]
            previous = self._buffers.copy()
            self._buffers.clear()
            for index in range(num_groups):
                name = str(index)
                saved = state_dict.get(f"{prefix}{name}")
                buffer = previous.get(name)
                if isinstance(saved, Tensor):
                    if type(saved) is not Tensor:
                        buffer = saved.to(
                            buffer.device
                            if buffer is not None
                            else saved.device
                        ).clone()
                        materialized.append(f"{prefix}{name}")
                        state_dict.pop(f"{prefix}{name}")
                    elif buffer is None:
                        buffer = torch.empty_like(saved)
                    elif (
                        buffer.shape != saved.shape
                        or buffer.dtype != saved.dtype
                    ):
                        buffer = buffer.new_empty(
                            saved.shape, dtype=saved.dtype
                        )
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
        for name in materialized:
            if name in missing_keys:
                missing_keys.remove(name)
        self._groups = tuple(
            cast(Tensor, group) for group in self._buffers.values()
        )
