# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=sp.StypeDispatch(
            numerical=[
                sp.ImputeMean(),
                sp.PCA(num_components=40),
                sp.MinMaxScale(),
            ]
        ),
    )
