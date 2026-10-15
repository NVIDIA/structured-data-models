# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kumo time-series model components."""

__lazy_modules__ = {
    "sdm.models.kumo.timeseries.forecasting",
}

from sdm.models.kumo.timeseries.forecasting import KumoForecast


__all__ = [
    "KumoForecast",
]
