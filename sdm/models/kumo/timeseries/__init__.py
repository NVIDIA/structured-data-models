# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kumo time-series model components."""

__lazy_modules__ = {
    "sdm.models.kumo.timeseries.forecast",
}

from sdm.models.kumo.timeseries.forecast import KumoForecast


__all__ = [
    "KumoForecast",
]
