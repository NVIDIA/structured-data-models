# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kumo model components."""

__lazy_modules__ = {
    "sdm.models.kumo.relational",
    "sdm.models.kumo.tabular",
}

from sdm.models.kumo.relational import KumoRelational
from sdm.models.kumo.tabular import KumoTabular


__all__ = [
    "KumoRelational",
    "KumoTabular",
]
