# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.Interpolate(),
            sp.Detrend(threshold=0.5),
            sp.Clip(-1e20, 1e20),
            # Negate every second estimator for symmetric averaging:
            sp.Choice(
                sp.Identity(),
                sp.FlipSign(probability=1.0),
                method="round_robin",
            ),
        ],
        target=[
            sp.Interpolate(),
            sp.Detrend(threshold=0.5),
            sp.Clip(-1e20, 1e20),
            sp.Choice(
                sp.Identity(),
                sp.FlipSign(probability=1.0),
                method="round_robin",
            ),
        ],
        output=sp.Sequential(
            sp.SortQuantiles(num_quantiles=9),
            sp.AverageEstimators(),
        ),
    )
