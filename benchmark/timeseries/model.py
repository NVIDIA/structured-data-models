# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass

import torch

import sdm


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: Callable[[torch.device], sdm.models.ICLModel]
    num_estimators: int = 1
    max_context_size: int | None = None
    max_variates: int | None = None


MODEL_CONFIGS = {
    "timesfm3": ModelConfig(
        name="TimesFM3",
        factory=lambda device: sdm.models.TimesFM3(
            accept_license=True,
            device=device,
        ),
        num_estimators=2,
        max_context_size=15_360,
        max_variates=32,
    ),
}
