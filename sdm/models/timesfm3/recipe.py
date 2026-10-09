# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=sp.LinearDetrend("__timesfm3_id__", threshold=0.5),
        target=sp.LinearDetrend("__timesfm3_id__", threshold=0.5),
        output=sp.AverageEstimators(),
    )
