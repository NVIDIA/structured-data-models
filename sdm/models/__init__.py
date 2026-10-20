# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured Data Models."""

__lazy_modules__ = {
    "sdm.models.tabiclv2",
    "sdm.models.tabfm",
    "sdm.models.kumo",
    "sdm.models.timesfm3",
}

from sdm.models.base import ICLModel, ECOC
from sdm.models.tabiclv2 import TabICLv2
from sdm.models.tabfm import TabFM
from sdm.models.kumo import KumoTabular, KumoRelational, KumoForecast
from sdm.models.timesfm3 import TimesFM3


__all__ = [
    "ICLModel",
    "ECOC",
    "TabICLv2",
    "TabFM",
    "KumoTabular",
    "KumoRelational",
    "KumoForecast",
    "TimesFM3",
]
