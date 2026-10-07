# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Picklable factories for real Kumo workers using prepared training data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import KumoRelational, KumoTabular


def _dtype(precision: str) -> torch.dtype | None:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": None,
        "float32": None,
    }[precision]


@dataclass(frozen=True)
class TabularProcessFactory:
    """Fit identical KumoTabular replicas from the runner's training arrays.

    This factory opens x_train.npy and y_train.npy only. Query arrays are sent
    as complete batches by the parent; validation targets are never opened.
    """

    data: Path
    context: int
    task: Literal["classification", "regression"] = "classification"
    size: Literal["small", "medium", "large"] = "small"
    estimators: int = 4
    seed: int = 1729
    precision: str = "bfloat16"
    estimator_batch_size: int | None = 1
    pretrained: bool = True

    def __call__(self, worker: int, device: torch.device) -> KumoTabular:
        """Construct weights and fit TRAIN context in the child process."""
        torch.manual_seed(self.seed)
        model = KumoTabular(
            task=self.task,
            size=self.size,
            pretrained=self.pretrained,
            device=device,
        )
        x = TableTensor.from_tensor(
            torch.from_numpy(
                np.load(self.data / "x_train.npy", mmap_mode="r")[
                    : self.context
                ].copy()
            )
            .float()
            .to(device)
        )
        values = (
            torch.from_numpy(
                np.load(self.data / "y_train.npy", mmap_mode="r")[
                    : self.context
                ].copy()
            )
            .reshape(-1, 1)
            .to(device)
        )
        y = (
            TableTensor(
                columns={Stype.categorical: ("target",)},
                categorical=CategoricalTensor.from_tensor(values.long()),
            )
            if self.task == "classification"
            else TableTensor.from_tensor(values.float())
        )
        dtype = _dtype(self.precision)
        with (
            torch.inference_mode(),
            torch.autocast(
                device.type,
                dtype=dtype,
                enabled=dtype is not None and device.type == "cuda",
            ),
        ):
            model.fit(
                x,
                y,
                num_estimators=self.estimators,
                estimator_batch_size=self.estimator_batch_size,
                generator=torch.Generator(device=device).manual_seed(
                    self.seed
                ),
            )
        return model


@dataclass(frozen=True)
class RelationalProcessFactory:
    """Fit identical KumoRelational replicas from trusted prepared graphs.pt.

    graphs.pt contains context and target-free queries; it must be the exact
    locally prepared artifact used by the single-process reference. No
    validation-label file is opened. Its objects require weights_only=False.
    """

    graphs: Path
    target: str
    task: Literal["classification", "regression"] = "classification"
    estimators: int = 4
    seed: int = 1729
    precision: str = "bfloat16"
    num_hops: int = 2
    estimator_batch_size: int | None = 1
    pretrained: bool = True

    def __call__(self, worker: int, device: torch.device) -> KumoRelational:
        """Construct weights and fit the exact prepared TRAIN graph."""
        torch.manual_seed(self.seed)
        context = torch.load(
            self.graphs,
            map_location="cpu",
            weights_only=False,
        )["context"]
        model = KumoRelational(
            task=self.task,
            pretrained=self.pretrained,
            device=device,
        )
        dtype = _dtype(self.precision)
        with (
            torch.inference_mode(),
            torch.autocast(
                device.type,
                dtype=dtype,
                enabled=dtype is not None and device.type == "cuda",
            ),
        ):
            model.fit(
                context.task_table.drop_columns(self.target).to(device),
                context.task_table[:, self.target].to(device),
                context.related_tables.to(device),
                num_estimators=self.estimators,
                num_hops=self.num_hops,
                estimator_batch_size=self.estimator_batch_size,
                generator=torch.Generator(device=device).manual_seed(
                    self.seed
                ),
            )
        return model
