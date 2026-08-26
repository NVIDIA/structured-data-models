from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.processing.execution import RecipeExecution

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


@dataclass(frozen=True)
class PlacementSummary:
    r"""Tensor storage owned by a compiled context, grouped by memory tier.

    Shared storage and tensor views are counted once. Model parameters are not
    part of a compiled context and are therefore excluded.

    Args:
        host_bytes: Bytes in pageable host memory.
        pinned_host_bytes: Bytes in pinned host memory.
        device_bytes_by_device: Bytes on each non-CPU device.
    """

    host_bytes: int
    pinned_host_bytes: int
    device_bytes_by_device: Mapping[torch.device, int]

    @property
    def total_bytes(self) -> int:
        r"""Total persistent tensor storage across all tiers."""
        return (
            self.host_bytes
            + self.pinned_host_bytes
            + sum(self.device_bytes_by_device.values())
        )


class CompiledContext:
    r"""Reusable fitted state for an :class:`~sdm.models.ICLModel`.

    A compiled context contains fitted recipe state and model intermediates,
    but never model parameters. Contexts are tied to the exact model instance
    and weights that created them. They may be retained independently
    and passed to :meth:`~sdm.models.ICLModel.predict_context` in any order.

    Prediction does not mutate a context. SDM does not serialize concurrent
    execution through one model, so callers must externally serialize
    predictions when the model implementation or device requires it.

    Multi-estimator CUDA compilation can intentionally retain estimator caches
    in pinned host memory; :attr:`placement` reports that mixed placement.

    Call :meth:`close` to release the context's fitted state deterministically.
    Closing is idempotent; prediction and placement inspection after closing
    fail.
    Compiled contexts are process-local and are not serializable.
    """

    def __init__(
        self,
        *,
        recipe_execution: RecipeExecution,
        cache: Cache,
        owner_token: object,
        weight_versions: tuple[tuple[object, ...], ...],
        submodel_token: object,
        task: Literal["classification", "regression"],
    ) -> None:
        self._recipe_execution: RecipeExecution | None = recipe_execution
        self._cache: Cache | None = cache
        self._owner_token = owner_token
        self._weight_versions = weight_versions
        self._submodel_token = submodel_token
        self._task = task

    @property
    def placement(self) -> PlacementSummary:
        r"""Persistent tensor storage grouped by host and device tier."""
        host_bytes = 0
        pinned_host_bytes = 0
        device_bytes: dict[torch.device, int] = {}
        seen: set[tuple[torch.device, int, int]] = set()
        for tensor in self._tensors():
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr(), storage.nbytes())
            if key in seen:
                continue
            seen.add(key)
            if tensor.device.type != "cpu":
                device_bytes[tensor.device] = (
                    device_bytes.get(tensor.device, 0) + storage.nbytes()
                )
            elif tensor.is_pinned():
                pinned_host_bytes += storage.nbytes()
            else:
                host_bytes += storage.nbytes()

        return PlacementSummary(
            host_bytes=host_bytes,
            pinned_host_bytes=pinned_host_bytes,
            device_bytes_by_device=device_bytes,
        )

    def close(self) -> None:
        r"""Release fitted recipe and model-cache state.

        This operation is idempotent. It does not release or modify the model's
        shared parameters and does not call ``torch.cuda.empty_cache()``.
        """
        self._recipe_execution = None
        self._cache = None

    def _state(self) -> tuple[RecipeExecution, Cache]:
        if self._recipe_execution is None or self._cache is None:
            raise RuntimeError("CompiledContext is closed")
        return self._recipe_execution, self._cache

    def _tensors(self) -> Iterator[Tensor]:
        recipe_execution, cache = self._state()
        yield from recipe_execution._tensors()
        yield from cache._tensors()

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError(
            "CompiledContext is process-local and not serializable"
        )
