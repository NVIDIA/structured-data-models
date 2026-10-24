# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning cache slices, optionally retaining bounded ICL replay replicas."""

from collections.abc import Hashable, Iterator
from math import prod
from typing import Any, Self, cast

from torch import Tensor

from sdm.cache import Cache, KVCacheEntry
from sdm.execution._collectives import Exchange


class OwnedCache(Cache):
    """Store columns for the frontend or context rows for ICL."""

    def __init__(
        self,
        exchange: Exchange,
        lengths: tuple[int, ...],
        dim: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.exchange, self.lengths = exchange, lengths
        self.dim, self.prefix = dim, prefix
        self.start = sum(lengths[: exchange.rank])
        self.length = lengths[exchange.rank]
        self._replicas: dict[str, KVCacheEntry] | None = None
        self._building_tensors: list[Tensor] | None = None

    def __setitem__(self, key: Hashable, value: object) -> None:
        if isinstance(key, str) and key.startswith(self.prefix):
            pair = cast(KVCacheEntry, value)
            value = KVCacheEntry(
                *(
                    tensor.narrow(self.dim, self.start, self.length)
                    .clone()
                    .contiguous()
                    for tensor in pair
                )
            )
        super().__setitem__(key, value)

    def __getitem__(self, key: Hashable) -> object:
        value = super().__getitem__(key)
        if (
            self.is_replaying
            and isinstance(key, str)
            and key.startswith(self.prefix)
        ):
            if self._replicas is not None:
                return self._replicas[key]
            pair = cast(KVCacheEntry, value)
            return KVCacheEntry(
                *(
                    self.exchange.gather(
                        tensor, self.lengths, self.dim
                    ).contiguous()
                    for tensor in pair
                )
            )
        return value

    def replica_plan(self) -> tuple[tuple[Any, ...], int]:
        """Return full-KV geometry and retained bytes on each rank."""
        layout: list[Any] = []
        total = 0
        for key, value in self._items.items():
            if not isinstance(key, str) or not key.startswith(self.prefix):
                continue
            pair = cast(KVCacheEntry, value)
            shapes = []
            for tensor in pair:
                shape = list(tensor.shape)
                shape[self.dim % tensor.ndim] = sum(self.lengths)
                shapes.append((tuple(shape), tensor.dtype))
                total += prod(shape) * tensor.element_size()
            layout.append((key, tuple(shapes)))
        return tuple(layout), total

    def materialize_replicas(self, layout: tuple[Any, ...]) -> None:
        """Gather whole ICL KV once, retaining partial payload on failure."""
        building: dict[str, KVCacheEntry] = {}
        building_tensors: list[Tensor] = []
        self._building_tensors = building_tensors
        for key, _ in layout:
            pair = self.exchange.attempt(
                lambda key=key: cast(
                    KVCacheEntry, Cache.__getitem__(self, key)
                )
            )
            full = []
            for tensor in pair:
                gathered = self.exchange.gather(tensor, self.lengths, self.dim)
                self.exchange.attempt(
                    lambda gathered=gathered, full=full: (
                        building_tensors.append(gathered),
                        full.append(gathered),
                    )
                )
            self.exchange.attempt(
                lambda key=key, full=full: building.__setitem__(
                    key, KVCacheEntry(*full)
                )
            )

        def install() -> None:
            self._replicas = building
            self._building_tensors = None

        self.exchange.attempt(install)

    def replica_state(self) -> tuple[Any, ...]:
        """Capture mapping identity beside tensor version identities."""
        replicas = self._replicas
        return (
            id(replicas),
            tuple((key, id(pair)) for key, pair in replicas.items())
            if replicas is not None
            else (),
        )

    def _tensors(self) -> Iterator[Tensor]:
        # Iteration over Cache.values() would gather replay entries.
        for value in self._items.values():
            if isinstance(value, KVCacheEntry):
                yield from value
            elif isinstance(value, Tensor):
                yield value
        if self._replicas is not None:
            for pair in self._replicas.values():
                yield from pair
        if self._building_tensors is not None:
            yield from self._building_tensors

    def freeze(self) -> Self:
        # Entries contain owning tensors only; do not invoke replay getters.
        self._mode = Cache.Mode.replay
        return self
