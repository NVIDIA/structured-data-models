# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Recipe
from sdm.models.kumo.timeseries.anomaly.preprocessing import AnomalyPreprocess


def default_recipe(
    num_features: int = 40, scale_factor: float = 1.0
) -> Recipe:
    """Build the feature-matching and min-max recipe for Kumo-Anomaly.

    Args:
        num_features: Checkpoint feature width.
        scale_factor: Multiplier applied after min-max normalization.
    """
    return Recipe(
        features=AnomalyPreprocess(
            num_features=num_features, scale_factor=scale_factor
        )
    )
