# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp
from sdm.models.tabiclv2.recipe import default_recipe as tabicl_recipe


def default_recipe() -> sp.Recipe:  # noqa: D103
    return tabicl_recipe().prepend_features(
        sp.TableDispatch(  # Add calendar features to related tables:
            related=sp.StypeDispatch(
                datetime=sp.AddCalendarFields(
                    ("minute", "hour", "weekday", "day_of_month", "month"),
                ),
            ),
        )
    )
