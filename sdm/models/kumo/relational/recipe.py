# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm
import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    # Add calendar features to related tables:
    return sdm.models.TabICLv2.default_recipe().prepend_features(
        sp.TableDispatch(
            related=sp.StypeDispatch(
                datetime=sp.AddCalendarFields(
                    ("minute", "hour", "weekday", "day_of_month", "month"),
                ),
            ),
        )
    )
