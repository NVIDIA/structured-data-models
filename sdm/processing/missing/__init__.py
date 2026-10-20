# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processors that mask numerical and categorical cells."""

from sdm.processing.missing.random import MaskRandom
from sdm.processing.missing.near_empty import MaskNearEmpty
from sdm.processing.missing.latent import MaskLatent
from sdm.processing.missing.structured import (
    MaskRows,
    MaskBlock,
    MaskByTarget,
    MaskCategorical,
    MaskCensoring,
)

__all__ = [
    "MaskRandom",
    "MaskNearEmpty",
    "MaskLatent",
    "MaskRows",
    "MaskBlock",
    "MaskByTarget",
    "MaskCategorical",
    "MaskCensoring",
]
