# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp

TIME_COLUMN = "__timesfm3_time__"


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=sp.LinearDetrend(TIME_COLUMN, threshold=0.5),
        target=sp.LinearDetrend(TIME_COLUMN, threshold=0.5),
        output=sp.AverageEstimators(),
    )
