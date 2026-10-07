# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin runner adapter for identical per-replica fit and whole-batch DP."""

from collections.abc import Sequence
from typing import Any

import torch
from research.multigpu.query_parallel import QueryBatch, QueryParallel

from sdm import RelatedTables, TableTensor
from sdm.models import ICLModel


class DataParallelAdapter:
    """Fit identical native replicas and dispatch complete prepared batches.

    Args:
        replicas: Models already loaded on their individual devices.
        dtype: Explicit worker autocast dtype, or None for FP32.
    """

    def __init__(
        self,
        replicas: Sequence[ICLModel],
        *,
        dtype: torch.dtype | None,
    ) -> None:
        self.replicas = tuple(replicas)
        self.devices = tuple(
            next(model.parameters()).device for model in replicas
        )
        self.dtype = dtype
        self.executor: QueryParallel | None = None

    def fit(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        generator: torch.Generator,
        **kwargs: Any,
    ) -> None:
        """Reproduce identical fitted recipes and model RNG on each replica."""
        if self.executor is not None:
            self.executor.close()
        for model, device in zip(self.replicas, self.devices, strict=True):
            if generator.device.type != device.type:
                raise ValueError(
                    "Fit generator must have the model device type"
                )
            local_generator = torch.Generator(device=device)
            local_generator.set_state(generator.get_state())
            with (
                torch.inference_mode(),
                torch.autocast(
                    device.type,
                    dtype=self.dtype,
                    enabled=self.dtype is not None and device.type == "cuda",
                ),
            ):
                model.fit(
                    x.to(device),
                    y.to(device),
                    None
                    if related_tables is None
                    else related_tables.to(device),
                    generator=local_generator,
                    **kwargs,
                )
        self.executor = QueryParallel(
            self.replicas, self.devices, dtype=self.dtype
        )

    def predict_batches(
        self,
        batches: Sequence[TableTensor | QueryBatch],
    ) -> list[TableTensor]:
        """Schedule the full pass once to permit inter-batch concurrency."""
        assert self.executor is not None
        prepared: list[QueryBatch] = []
        offset = 0
        for batch in batches:
            if isinstance(batch, QueryBatch):
                prepared.append(batch)
            else:
                prepared.append(
                    QueryBatch(
                        tuple(range(offset, offset + batch.size(-2))),
                        batch.to("cpu"),
                    )
                )
                offset += batch.size(-2)
        return [
            result.prediction for result in self.executor.predict(prepared)
        ]

    def predict(
        self,
        x: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> TableTensor:
        """Run one indivisible batch; DP cannot accelerate this operation."""
        return self.predict_batches(
            [
                QueryBatch(
                    tuple(range(x.size(-2))),
                    x.to("cpu"),
                    None
                    if related_tables is None
                    else related_tables.to("cpu"),
                )
            ]
        )[0]

    def close(self) -> None:
        """Drain pending work and release inference threads."""
        if self.executor is not None:
            self.executor.close()


def factory(args: Any, replicas: Sequence[ICLModel]) -> DataParallelAdapter:
    """Create the benchmark adapter from runner precision and replicas."""
    precision = args.precision
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": None,
        "float32": None,
    }[precision]
    return DataParallelAdapter(replicas, dtype=dtype)
