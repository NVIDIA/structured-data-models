# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    # Frozen statistics from standardizer.pkl in nvidia/Kumo-Forecast:
    mean = -5.671202659606934
    scale = 8.693312644958496

    return sp.Recipe(
        features=sp.Standardize().with_state(mean, scale),
        target=sp.Standardize().with_state(mean, scale),
        output=sp.AverageEstimators(),
    )
