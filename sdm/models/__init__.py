# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data models."""

__lazy_modules__ = {
    "sdm.models.tabiclv2",
    "sdm.models.tabfm",
    "sdm.models.kumo",
    "sdm.models.timesfm3",
}

from sdm.models.base import Callback, Model, ICLModel, ECOC
from sdm.models.tabiclv2 import TabICLv2
from sdm.models.tabfm import TabFM
from sdm.models.kumo import (
    KumoTabular,
    KumoRelational,
    KumoForecast,
    KumoAnomaly,
)
from sdm.models.timesfm3 import TimesFM3


__all__ = [
    "Callback",
    "Model",
    "ICLModel",
    "ECOC",
    "TabICLv2",
    "TabFM",
    "KumoTabular",
    "KumoRelational",
    "KumoForecast",
    "KumoAnomaly",
    "TimesFM3",
]
