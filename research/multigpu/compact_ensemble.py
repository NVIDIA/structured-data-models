# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact cache views retaining unused projection storage after fit."""

from __future__ import annotations

import time
from argparse import Namespace
from collections.abc import Sequence
from functools import partial
from typing import Any

import torch

from sdm.cache import Cache
from sdm.models import EnsembleParallel, ICLModel


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        _storage_key(tensor),
        tensor.dtype,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def storage_bytes(cache: Cache) -> int:
    """Count unique backing allocations, including unused bytes in views."""
    storages = {
        _storage_key(tensor): tensor.untyped_storage().nbytes()
        for tensor in cache._tensors()
    }
    return sum(storages.values())


def compact_cache(cache: Cache) -> Cache:
    """Clone only views whose storage exceeds all referenced view bytes.

    Identical aliases are counted once and reuse one clone. Storage whose
    referenced views cover its full byte count is left intact. Overlapping
    different views can conservatively prevent compaction; values never change.
    """
    views = {_view_key(tensor): tensor for tensor in cache._tensors()}
    referenced: dict[tuple[str, int], int] = {}
    for tensor in views.values():
        key = _storage_key(tensor)
        referenced[key] = (
            referenced.get(key, 0) + tensor.numel() * tensor.element_size()
        )
    replacements = {
        key: tensor.clone(memory_format=torch.contiguous_format)
        for key, tensor in views.items()
        if referenced[_storage_key(tensor)] < tensor.untyped_storage().nbytes()
    }
    return cache._apply_tensor(
        lambda tensor: replacements.get(_view_key(tensor), tensor)
    )


class CompactEnsembleParallel(EnsembleParallel):
    """Compact independently fitted resident caches on their owning devices."""

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Fit ordinary resident members, then measure cache compaction."""
        super().fit(*args, **kwargs)

        def compact(
            i: int, model: ICLModel, device: torch.device
        ) -> tuple[int, int]:
            before = storage_bytes(self._caches[i])
            self._caches[i] = compact_cache(self._caches[i])
            return before, storage_bytes(self._caches[i])

        start = time.perf_counter()
        self.compaction_storage_bytes = self._dispatch(
            compact, len(self._caches), self.devices[0]
        )
        self.compaction_s = time.perf_counter() - start


def factory(
    args: Namespace, replicas: Sequence[ICLModel]
) -> CompactEnsembleParallel:
    """Build the compact-cache arm for the shared tabular benchmark."""
    model = CompactEnsembleParallel(replicas)
    model.fit = partial(model.fit, member_seed=args.seed)
    return model
