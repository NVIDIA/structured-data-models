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
    autocast_dtype: torch.dtype
    num_estimators: int = 1
    cross_learning: bool = False
    max_context_size: int | None = None
    max_variates: int | None = None


MODEL_CONFIGS = {
    "chronos2": ModelConfig(
        name="Chronos2",
        factory=lambda device: sdm.models.Chronos2(device=device),
        autocast_dtype=torch.float16,
        num_estimators=1,
        max_context_size=8_192,
        max_variates=None,
    ),
    "timesfm3": ModelConfig(
        name="TimesFM3",
        factory=lambda device: sdm.models.TimesFM3(
            accept_license=True,
            device=device,
        ),
        autocast_dtype=torch.bfloat16,
        num_estimators=2,
        max_context_size=15_360,
        max_variates=32,
    ),
}
