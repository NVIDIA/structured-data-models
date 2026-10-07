# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve worker autocast nesting across fixed-precision calls."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, TypeVar

import torch

from sdm.models import EnsembleParallel, ICLModel

T = TypeVar("T")


class PersistentAutocastEnsembleParallel(EnsembleParallel):
    """Hold one outer autocast scope on each persistent worker thread.

    Args:
        replicas: Identical evaluation replicas on distinct devices.
        dtype: Fixed autocast dtype, or None for unautocast FP32.

    Call fit/predict with this same autocast setting. Different settings are
    rejected to avoid reusing cached casts across incompatible precisions.
    Replica weights must remain unchanged until close.
    """

    def __init__(
        self, replicas: Sequence[ICLModel], *, dtype: torch.dtype | None
    ) -> None:
        super().__init__(replicas)
        self._dtype = dtype

        def enter(device: torch.device) -> tuple[Any, Any]:
            inference = torch.inference_mode()
            autocast = torch.autocast(
                device.type, dtype=dtype, enabled=dtype is not None
            )
            inference.__enter__()
            autocast.__enter__()
            return inference, autocast

        futures = [
            worker.submit(enter, device)
            for worker, device in zip(self._workers, self.devices, strict=True)
        ]
        self._contexts = [future.result() for future in futures]

    def _dispatch(
        self,
        operation: Callable[[int, ICLModel, torch.device], T],
        count: int,
        source: torch.device,
    ) -> list[T]:
        for device in self.devices:
            enabled = torch.is_autocast_enabled(device.type)
            if enabled != (self._dtype is not None) or (
                enabled
                and torch.get_autocast_dtype(device.type) != self._dtype
            ):
                raise ValueError(
                    "Caller autocast must match the fixed worker precision"
                )
        return super()._dispatch(operation, count, source)

    def close(self) -> None:
        """Exit autocast on its owning threads, then join the workers."""
        if self._contexts:

            def leave(contexts: tuple[Any, Any]) -> None:
                inference, autocast = contexts
                autocast.__exit__(None, None, None)
                inference.__exit__(None, None, None)

            futures = [
                worker.submit(leave, contexts)
                for worker, contexts in zip(
                    self._workers, self._contexts, strict=True
                )
            ]
            for future in futures:
                future.result()
            self._contexts = []
        super().close()


def factory(
    args: Namespace, replicas: Sequence[ICLModel]
) -> PersistentAutocastEnsembleParallel:
    """Build the fixed-precision lifetime experiment for tabular_bench."""
    model = PersistentAutocastEnsembleParallel(
        replicas,
        dtype=torch.bfloat16 if args.precision == "bfloat16" else None,
    )
    model.fit = partial(model.fit, member_seed=args.seed)
    return model
