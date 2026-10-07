# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin runner adapter for identical per-replica fit and whole-batch DP."""

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from research.multigpu.query_parallel import QueryBatch, QueryParallel

from sdm import RelatedTables, TableTensor
from sdm.models import ICLModel


class FittablePredictor(Protocol):
    """Model or ensemble group accepted by the common benchmark adapter."""

    def fit(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit a complete identical context and recipe."""
        ...

    def predict(
        self, x: TableTensor, related_tables: RelatedTables | None = None
    ) -> TableTensor:
        """Predict one complete query batch."""
        ...


class DataParallelAdapter:
    """Fit identical native replicas and dispatch complete prepared batches.

    Args:
        replicas: Models or ensemble groups already loaded on their devices.
        devices: Input device for each replica or ensemble group.
        dtype: Explicit worker autocast dtype, or None for FP32.
        autocast_device_type: Compute backend if different from input device.
        estimator_batch_size: Same within-replica estimator batching as the
            native reference, unless explicitly overridden by fit.
    """

    def __init__(
        self,
        replicas: Sequence[FittablePredictor],
        *,
        devices: Sequence[torch.device],
        dtype: torch.dtype | None,
        estimator_batch_size: int | None = 1,
        member_seed: int | None = None,
        autocast_device_type: str | None = None,
    ) -> None:
        self.replicas = tuple(replicas)
        self.devices = tuple(devices)
        self.dtype = dtype
        self.estimator_batch_size = estimator_batch_size
        self.member_seed = member_seed
        self.autocast_device_type = autocast_device_type
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
        self.executor = None
        if self.member_seed is None:
            kwargs.setdefault(
                "estimator_batch_size", self.estimator_batch_size
            )
        else:
            if kwargs.pop("estimator_batch_size", 1) != 1:
                raise ValueError(
                    "Hybrid ensemble groups run members separately"
                )
            kwargs.setdefault("member_seed", self.member_seed)
        for model, device in zip(self.replicas, self.devices, strict=True):
            if generator.device.type != device.type:
                raise ValueError(
                    "Fit generator must have the model device type"
                )
            local_generator = torch.Generator(device=device)
            local_generator.set_state(generator.get_state())
            autocast_type = self.autocast_device_type or device.type
            with (
                torch.inference_mode(),
                torch.autocast(
                    autocast_type,
                    dtype=self.dtype,
                    enabled=self.dtype is not None and autocast_type == "cuda",
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
            self.replicas,
            self.devices,
            dtype=self.dtype,
            autocast_device_type=self.autocast_device_type,
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
    return DataParallelAdapter(
        replicas,
        devices=[next(replica.parameters()).device for replica in replicas],
        dtype=dtype,
        estimator_batch_size=getattr(args, "estimator_batch_size", 1),
    )


def hybrid_factory(
    args: Any, replicas: Sequence[ICLModel]
) -> DataParallelAdapter:
    """Compose two DP groups, each using half the replicas for ensemble work.

    Requires the separately implemented EnsembleParallel prototype. Compare
    with its one-GPU resident/member-seeded reference rather than only native.
    """
    from sdm.models.ensemble_parallel import EnsembleParallel  # noqa: PLC0415

    if len(replicas) < 2 or len(replicas) % 2:
        raise ValueError("Hybrid requires two equally sized ensemble groups")
    width = len(replicas) // 2
    groups = [
        EnsembleParallel(replicas[:width]),
        EnsembleParallel(replicas[width:]),
    ]
    base = factory(args, replicas)
    cpu_recipe = getattr(args, "recipe_device", "cuda") == "cpu"

    class HybridAdapter(DataParallelAdapter):
        def close(self) -> None:
            super().close()
            for group in groups:
                group.close()

    return HybridAdapter(
        groups,
        devices=(
            [torch.device("cpu"), torch.device("cpu")]
            if cpu_recipe
            else [base.devices[0], base.devices[width]]
        ),
        dtype=base.dtype,
        member_seed=args.seed,
        autocast_device_type=base.devices[0].type,
    )
